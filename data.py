# data.py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class DrivingDatasetHWC(Dataset):
    """
    Returns:
      img:  FloatTensor (H, W, C) in 0..255, dtype=float32  [matches old code]
      y:    class index (LongTensor scalar) OR raw action (FloatTensor)
    Notes:
      - If augment=True, we apply label-aware horizontal flip and ColorJitter.
      - ColorJitter is applied in CHW on [0,1] then scaled back to 0..255 HWC.
    """
    def __init__(self, obs_path, actions_array, to_class_fn,
                 train: bool, augment: bool):
        # Memory-map observations to avoid loading all into RAM
        self.obs = np.load(obs_path, mmap_mode="r")  # expect (N,H,W,C)
        assert self.obs.ndim == 4 and self.obs.shape[-1] == 3, "observations must be (N,H,W,3)"
        self.actions_np = actions_array               # numpy array (N, A)
        self.to_class = to_class_fn                   # callable: action -> class index (int)
        self.train = train
        self.augment = augment

        # Photometric jitter (brightness/contrast/saturation optional)
        self.jitter = T.ColorJitter(brightness=0.5)

    def __len__(self):
        return self.obs.shape[0]

    def _photometric(self, img_hwc_f32_255):
        """Apply ColorJitter by converting to CHW [0,1], then back to HWC 0..255."""
        x_chw = img_hwc_f32_255.permute(2, 0, 1) / 255.0          # (C,H,W) in [0,1]
        x_chw = self.jitter(x_chw)                                 # torchvision op
        x_hwc = (x_chw.clamp(0, 1) * 255.0).permute(1, 2, 0)       # back to HWC 0..255
        return x_hwc

    def __getitem__(self, i):
        # img (H,W,C) as float32 in 0..255, no normalization
        img = torch.from_numpy(self.obs[i]).to(torch.float32)
        act = torch.from_numpy(self.actions_np[i]).to(torch.float32)

        flipped = False
        if self.train and self.augment:
            # Random horizontal flip with p=0.5 (label-aware)
            if random.random() < 0.5:
                img = TF.hflip(img)           # works on HWC
                act = act.clone()
                act[0] = -act[0]              # invert steering
                flipped = True

            # Photometric jitter
            img = self._photometric(img)

        # Turn action into class AFTER any label-space change (e.g., flip)
        # to_class returns an integer class index
        y = self.to_class(act)

        # Return HWC 0..255 and class index
        return img, torch.as_tensor(y, dtype=torch.long)
