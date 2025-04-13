
from datetime import datetime
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from ultralytics.tracknet.utils.nms import non_max_suppression
from ultralytics.tracknet.utils.transform import revert_coordinates
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.engine.predictor import STREAM_WARNING, BasePredictor
from ultralytics.yolo.engine.results import Results

from ultralytics.yolo.utils import LOGGER, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import select_device
import platform
from pathlib import Path
import cv2
from dataclasses import dataclass
from typing import Optional
import psutil
import pynvml


@dataclass
class Prediction:
    x: float
    y: float
    conf: float

@dataclass
class ResultItem:
    pred: Prediction
    speed: dict[str, float | None]

class TrackNetPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, pynvml.NVMLError_LibraryNotFound):
            print("⚠️ NVML not available on this system (probably no NVIDIA GPU).")
            self.gpu_available = False
        self.proc = psutil.Process(os.getpid())

    def profile_resources(self, tag=""):
        cpu = self.proc.cpu_percent(interval=None)
        mem = self.proc.memory_info().rss / 1024**2
        if self.gpu_available:
            torch.cuda.synchronize()
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            print(f"[{tag}] CPU: {cpu:.1f}%, RAM: {mem:.1f}MB, GPU: {util.gpu}%, vRAM: {mem_info.used/1024**2:.1f}MB")
        else:
            print(f"[{tag}] CPU: {cpu:.1f}%, RAM: {mem:.1f}MB (No GPU available)")

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
    # def setup_model(self, model, verbose=True):
    #     """Initialize YOLO model with given parameters and set it to evaluation mode."""
    #     self.model = model
    #     self.device = select_device(self.args.device, verbose=verbose)  # update device
    #     self.args.half = True  # update half
    #     # self.args.half = self.args.half  # update half
    #     self.model.eval()
    def preprocess(self, im):
        self.profile_resources("Preprocess (before)")
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = im.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        assert im.ndim == 3 and im.shape[0] == 10, "Expect shape (10, H, W)"
        img = im.to(self.device)

        img = img.float()  # 若 im 是 uint8，轉為 float32
        median = img.median(dim=0).values  # shape: (H, W)
        img = img - median
        img = torch.clamp(img, 0, 255)

        # Normalize
        img /= 255.0

        # Output shape: (1, 10, 640, 640)
        result = img.unsqueeze(0).to(self.device).half() if self.model.fp16 else img.unsqueeze(0).to(self.device)
        self.profile_resources("Preprocess (after)")
        return result

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        use_nms = True
        conf_threshold = 0.5
        nc = 1
        reg_max = 16
        feat_no = 8
        no = nc + reg_max * feat_no
        cell_num = 80
        stride = 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj = torch.arange(reg_max, dtype=torch.float, device=device)

        feats = preds[0].clone()
        pred_distri, pred_scores = feats.view(no, -1).split(
            (reg_max * feat_no, nc), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*self.cell_num*self.cell_num]
        
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, feat_no, c // feat_no).softmax(2).matmul(
            proj.type(pred_distri.dtype))
        each_probs = pred_probs.view(10, cell_num, cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, cell_num, cell_num, feat_no).split([2, 2, 2, 2], dim=3)

        orig_images_clone = orig_imgs.transpose(2, 0, 1)

        p = Path(self.batch[0][0])
        parent_dir = p.parent.name
        save_path = os.path.join(self.save_dir, parent_dir)
        os.makedirs(save_path, exist_ok=True)
        result = []
        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            center = 0.5

            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            frame_preds = []
            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    frame_preds.append(ResultItem(
                        pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                        speed={'preprocess': None, 'inference': None, 'postprocess': None }
                    ))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                frame_preds.append(ResultItem(
                    pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                    speed={'preprocess': None, 'inference': None, 'postprocess': None }
                ))
            
            result.append(ResultItem(
                pred=frame_preds if use_nms else frame_preds[0],
                speed={'preprocess': None, 'inference': None, 'postprocess': None}
            ))
            
            # 視覺化與儲存圖片
            img_np = orig_images_clone[frame_idx, :, :]
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            img_np = np.ascontiguousarray(img_np.copy())

            for frame_pred in frame_preds:
                pred = frame_pred.pred
                if pred.conf >= conf_threshold:
                    cv2.circle(img_np, (int(pred.x.item()), int(pred.y.item())), radius=3, color=(0, 0, 255), thickness=-1)
                    conf_text = f"{pred.conf:.2f}"
                    cv2.putText(img_np, conf_text, (int(pred.x.item()) + 5, int(pred.y.item()) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)


            # 儲存圖片
            idx_p = Path(self.batch[0][frame_idx])
            cv2.imwrite(f"{save_path}/{idx_p.name}", img_np)
        
        # TODO: 這裡需要將結果轉換為原始圖片的座標系統
        # result = revert_coordinates(result, orig_imgs[0].shape[2], orig_imgs[0].shape[3], img[0].shape[2])
        return result
    def write_results(self, idx, results, batch):
        return "todo"
    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        return "todo"

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = '.mp4'
                fourcc = 'avc1'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)