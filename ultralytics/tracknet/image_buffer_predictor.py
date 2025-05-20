from datetime import time
from queue import Empty, Queue
import queue
import threading

import cv2
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.tracknet.protocal.image_buffer import FrameProtocol, ImageBufferProtocol
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, callbacks
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
import paho.mqtt.client as mqtt
from typing import Tuple, List

class ImageBufferPredictor:

    def __init__(self, weight: str, image_buffer: ImageBufferProtocol, output_width:int=None, output_height:int=None,
                 mqttc:mqtt.Client=None, output_topic:str=None,
                 cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = self.get_save_dir()
        verbose = False
        # Usable if setup is done
        self.model = AutoBackend(weight,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

        self.output_width = output_width
        self.output_height = output_height
        self.mqttc = mqttc
        self.output_topic = output_topic
        self.image_buffer = image_buffer
        self.track_size = 10
        self.imgsz = 640
        self.max_streams = 6
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.max_streams)]
        self.stream_idx = 0
        self.lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.event_q = queue.Queue()
        self.infer_q = queue.Queue()

        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

        
    def start(self):
        self.running = True
        threading.Thread(target=self._preprocess_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        threading.Thread(target=self._postprocess_loop, daemon=True).start()

    def stop(self):
        self.running = False
    
    def _preprocess_loop(self):
        while self.running:
            tensor, fids, timestamps = self._preprocess()
            self.stream_idx = (self.stream_idx + 1) % self.max_streams
            self.infer_q.put((tensor, (fids, timestamps), self.stream_idx))

    def _preprocess(self) -> Tuple[torch.Tensor, List[int], List[float]]:
        frames = []
        fids = []
        timestamps = []

        while len(frames) < self.track_size:
            frame = self.image_buffer.pop(True)
            if frame.is_eos:
                self.stop()
                
            img = frame.image.astype(np.float32)
            img = self.pad_to_square(img)
            img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=0)  # (1, H, W)
            frames.append(img)
            fids.append(frame.index)
            timestamps.append(frame.monotonic_timestamp)

        img = np.concatenate(frames, 0)
        img_tensor = torch.from_numpy(img).float().contiguous().pin_memory()

        return (img_tensor, fids, timestamps)

    def _inference_loop(self):
        while self.running:
            try:
                tensor, meta, stream_id = self.infer_q.get(timeout=0.1)
            except queue.Empty:
                continue
            stream = self.streams[stream_id]
            event = torch.cuda.Event()

            with torch.cuda.stream(stream):
                input_gpu = tensor.to(self.device, dtype=torch.float32, non_blocking=True)
                median = input_gpu.median(dim=1).values  # shape: (H, W)
                input_gpu.sub_(median).clamp_(0, 255).div_(255.0)

                if self.model.fp16:
                    input_gpu = input_gpu.half()

                with torch.no_grad():
                    output = self.model(input_gpu)
                event.record(stream)
            with self.event_lock:
                self.event_q.put((event, output, meta))

    def _postprocess_loop(self):
        while self.running:
            completed = []
            with self.event_lock:
                items = list(self.event_q.queue)
            for item in items:
                event, output, meta = item
                if event.query():
                    completed.append(item)
            for item in completed:
                with self.event_lock:
                    self.event_q.queue.remove(item)
                self.on_result(item[1], item[2])
            time.sleep(0.001)
    def on_result(self, output_tensor: torch.Tensor, meta):
        """
        使用者可 override 本函式。
        預設顯示 output.shape 與 meta。
        """
        print("[Result] output shape:", output_tensor.shape, "meta:", meta)