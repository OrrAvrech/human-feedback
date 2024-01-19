import torch
from torch.utils import data
from pathlib import Path
from typing import Optional
import numpy as np


class BaseMotion(data.Dataset):
    def __init__(self, dataset_dir: Path, max_frames: Optional[int] = None):
        self.dataset_dir = dataset_dir
        self.motions_list = list(self.dataset_dir.rglob("*.npy"))
        self.max_frames = max_frames

    def __getitem__(self, idx) -> np.array:
        filepath = self.motions_list[idx]
        motion = torch.Tensor(np.load(filepath))[:self.max_frames, ...]
        return motion

    def __len__(self):
        return len(self.motions_list)
    

class BaseMotionSplit(BaseMotion):
    def __init__(self, dataset_dir: Path, motion_dir: str, split: str, max_frames: Optional[int] = None):
        self.motion_dir = dataset_dir / motion_dir
        super().__init__(self.motion_dir, max_frames)
        self.split_file = dataset_dir / f"{split}.txt"
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.motions_list = [p for p in self.motions_list if p.stem in id_list]
    