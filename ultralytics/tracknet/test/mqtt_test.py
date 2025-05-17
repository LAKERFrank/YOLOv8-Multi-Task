import numpy as np
import torch
from torch.utils.data import DataLoader
import threading
import time
import gc
import psutil
import os
import cv2
from torch.utils.data import Dataset


class Frame:
    is_eos: bool
    def __init__(self) -> None:
        """__init__(self: recorder_module.Frame) -> None"""
    @property
    def height(self) -> int: ...
    @property
    def image(self) -> np.ndarray[np.uint8]: ...
    @property
    def index(self) -> int: ...
    @property
    def monotonic_timestamp(self) -> float: ...
    @property
    def timestamp(self) -> float: ...
    @property
    def width(self) -> int: ...

class ImageBuffer:
    def __init__(self) -> None:
        """__init__(self: recorder_module.ImageBuffer) -> None"""
    def clear(self) -> None:
        """clear(self: recorder_module.ImageBuffer) -> None"""
    def pop(self, blocking: bool = ...) -> Frame:
        """pop(self: recorder_module.ImageBuffer, blocking: bool = True) -> recorder_module.Frame"""
    def push(self, arg0: Frame) -> None:
        """push(self: recorder_module.ImageBuffer, arg0: recorder_module.Frame) -> None"""

class ImageBufferDataset(Dataset):
    def __init__(self, image_buffer: ImageBuffer, track_size: int = 10):
        self.image_buffer = image_buffer
        self.track_size = track_size
        self.buffer = []  # 儲存一組 track_size 連續 frame
        self.imgsz = 640

    def __len__(self):
        return 1_000_000  # 任意大，會由外部控制終止（如遇到 EOS）
    
    def pad_to_square(self, img, pad_value=0):
        h, w = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if h > w else (pad1, pad2, 0, 0)
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=pad_value)
        return img
    
    def __getitem__(self, index):
        frames = []
        # fids = []
        # timestamps = []

        while len(frames) < self.track_size:
            frame = self.image_buffer.pop(True)
            if frame.is_eos:
                raise StopIteration  # 結束迴圈
            img = frame.image.astype(np.float32)
            img = self.pad_to_square(img)
            img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=0)  # (1, H, W)
            frames.append(img)
            # frames.append(frame.image)
            # fids.append(frame.index)
            # timestamps.append(frame.monotonic_timestamp)

        img = np.concatenate(frames, 0)
        img_tensor = torch.from_numpy(img).float()
            
        return (f"stream_dataset_{index}", img_tensor, "", "")

# ==== 假的 Frame & ImageBuffer 實作 ====

class FakeFrame:
    def __init__(self, image: np.ndarray, index: int, is_eos=False):
        self._image = image
        self._index = index
        self._is_eos = is_eos
        self._timestamp = time.time()
        self._monotonic_timestamp = time.monotonic()

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_eos(self) -> bool:
        return self._is_eos

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def monotonic_timestamp(self) -> float:
        return self._monotonic_timestamp

    @property
    def height(self) -> int:
        return self._image.shape[0]

    @property
    def width(self) -> int:
        return self._image.shape[1]

class FakeImageBuffer:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

    def push(self, frame):
        with self.cv:
            self.queue.append(frame)
            self.cv.notify()

    def pop(self, blocking=True):
        with self.cv:
            while blocking and not self.queue:
                self.cv.wait()
            return self.queue.pop(0)

    def clear(self):
        with self.cv:
            self.queue.clear()

# ==== 引入你的 Dataset 實作 ====
# 假設 `ImageBufferDataset` 已經在前面定義

# ==== 測試程式 ====
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def producer(buffer: FakeImageBuffer, total_frames=1000, img_shape=(480, 640)):
    for i in range(total_frames):
        fake_image = np.random.randint(0, 256, img_shape, dtype=np.uint8)
        frame = FakeFrame(fake_image, index=i)
        buffer.push(frame)
        time.sleep(0.001)  # 模擬 stream

def run_test():
    fake_buffer = FakeImageBuffer()
    dataset = ImageBufferDataset(fake_buffer, track_size=10)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    prod_thread = threading.Thread(target=producer, args=(fake_buffer,))
    prod_thread.start()

    print("Start consuming...")
    try:
        for i, batch in enumerate(dataloader):
            print(f"Step {i} | Memory: {memory_usage():.2f} MB")
            del batch  # 主動釋放
            gc.collect()
            if i >= 50:
                break
    except StopIteration:
        print("Stream ended.")
    finally:
        prod_thread.join()

if __name__ == "__main__":
    run_test()
