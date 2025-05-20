from pathlib import Path
import threading
import time
import queue
from typing import Tuple, List

import cv2
import numpy as np
import torch
import paho.mqtt.client as mqtt

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.tracknet.protocal.image_buffer import FrameProtocol, ImageBufferProtocol

class ImageBufferPredictor:
    def __init__(self, weight: str, image_buffer: ImageBufferProtocol, output_width: int = None,
                 output_height: int = None, mqttc: mqtt.Client = None, output_topic: str = None,
                 cfg=DEFAULT_CFG, overrides=None):

        self.args = get_cfg(cfg, overrides)
        # self.save_dir = self.get_save_dir()
        self.model = AutoBackend(weight,
                                 device=select_device(self.args.device, verbose=False),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=False)
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()

        self.output_width = output_width
        self.output_height = output_height
        self.mqttc = mqttc
        self.output_topic = output_topic
        self.image_buffer = image_buffer
        self.track_size = 10
        self.imgsz = 640

        self.max_streams = torch.cuda.device_count() * 2
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.max_streams)]
        self.event_pool = queue.SimpleQueue()

        for _ in range(128):
            self.event_pool.put(torch.cuda.Event())

        self.stream_idx = 0
        self.infer_q = queue.Queue(maxsize=128)
        self.result_q = queue.Queue(maxsize=256)


    def get_save_dir(self):
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
    
    def start(self):
        self.running = True
        self.threads = [
            threading.Thread(target=self._preprocess_loop),
            threading.Thread(target=self._inference_loop),
            threading.Thread(target=self._postprocess_loop),
        ]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()  # 阻塞直到所有 thread 結束

    def stop(self):
        self.running = False

    def _preprocess_loop(self):
        while self.running:
            try:
                tensor, fids, timestamps = self._preprocess()
                self.stream_idx = (self.stream_idx + 1) % self.max_streams
                self.infer_q.put((tensor, (fids, timestamps), self.stream_idx), timeout=1)
            except Exception as e:
                LOGGER.warning(f"Preprocess loop error: {e}")

    def _preprocess(self) -> Tuple[torch.Tensor, List[int], List[float]]:
        frames, fids, timestamps = [], [], []
        while len(frames) < self.track_size:
            frame = self.image_buffer.pop(True)
            if frame.is_eos:
                self.stop()
                break

            img = frame.image.astype(np.float32)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.pad_to_square(img)
            img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            frames.append(np.expand_dims(img, axis=0))
            fids.append(frame.index)
            timestamps.append(frame.monotonic_timestamp)

        img = np.concatenate(frames, 0)
        return torch.from_numpy(img).contiguous().pin_memory(), fids, timestamps

    def _inference_loop(self):
        while self.running:
            try:
                tensor, meta, stream_id = self.infer_q.get(timeout=0.1)
                stream = self.streams[stream_id]
                event = self.event_pool.get()

                with torch.cuda.stream(stream):
                    input_gpu = tensor.to(self.device, non_blocking=True)
                    median = input_gpu.median(dim=0).values
                    input_gpu.sub_(median).clamp_(0, 255).div_(255.0)
                    if self.model.fp16:
                        input_gpu = input_gpu.half()
                    with torch.no_grad():
                        output = self.model(input_gpu)
                    event.record(stream)

                self.result_q.put((event, output, meta, stream))
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.warning(f"Inference loop error: {e}")

    def _postprocess_loop(self):
        while self.running:
            try:
                event, output, meta, stream = self.result_q.get(timeout=0.1)
                if event.query():
                    self.on_result(output, meta)
                    self.event_pool.put(event)
                else:
                    # 沒完成的重新放回，但避免 busy loop
                    self.result_q.put((event, output, meta, stream))
                    time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                LOGGER.warning(f"Postprocess loop error: {e}")


    def pad_to_square(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size), dtype=img.dtype)
        padded[:h, :w] = img
        return padded

    def on_result(self, output_tensor: torch.Tensor, meta):
        print("[Result] output shape:", output_tensor.shape, "meta:", meta, "fid", meta[0], "timestamp", meta[1], "endTime", time.time())   
