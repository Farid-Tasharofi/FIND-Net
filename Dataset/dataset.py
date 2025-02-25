import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import PIL
from PIL import Image
import PIL.Image
from gecatsim.pyfiles.CommonTools import *


def image_get_minmax():
    return 0.0, 1.0


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2.0 - 1.0
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        return img
    return [_augment(a) for a in args]


def save_image(img, file_path, i):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    file_name = os.path.join(file_path, f'output_image_{i}.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def check_nan_inf(data, name, absdir):
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"NaN or Inf found in {name}")
        print(absdir)
        exit()

def save_image_2d(img, file_path, i):
    import matplotlib.pyplot as plt
    # Assume img is a tensor of shape [batch_size, channels, height, width]
    # Plot each image in the batch
    plt.figure()
 
    # Save the image
    plt.axis('off')
    file_name = os.path.join(file_path, f'output_image_{i}.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_image_and_histogram(img, file_path, i):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Ensure directory exists
    os.makedirs(file_path, exist_ok=True)

    # Save the image
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    image_filename = os.path.join(file_path, f'output_image_{i}.png')
    plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Calculate and save the histogram
    plt.figure()
    # Flatten the image array and calculate the histogram
    histogram, bin_edges = np.histogram(img.flatten(), bins=256, range=[0,256])
    plt.fill_between(bin_edges[:-1], histogram, step="pre", alpha=0.75)
    plt.title('Grayscale Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    histogram_filename = os.path.join(file_path, f'histogram_{i}.png')
    plt.savefig(histogram_filename, bbox_inches='tight', pad_inches=0)
    plt.close()


class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize, length, mode='train', augment_mode = True, output_size=(512,512)):
        super().__init__()
        self.dir = dir
        self.patch_size = patchSize
        self.sample_num = length
        self.mode = mode
        self.augment = augment_mode
        self.output_size = output_size

        # Define directories for training and validation
        if mode == 'train':
            self.Xgt_path = os.path.join(self.dir, 'train/Target/')
            self.Xma_path = os.path.join(self.dir, 'train/Baseline/')
            self.XLI_path = os.path.join(self.dir, 'train/LI/')
            self.Mask_path = os.path.join(self.dir, 'train/Mask/')
        elif mode == 'test':
            self.Xgt_path = os.path.join(self.dir, 'test/Target/')
            self.Xma_path = os.path.join(self.dir, 'test/Baseline/')
            self.XLI_path = os.path.join(self.dir, 'test/LI/')
            self.Mask_path = os.path.join(self.dir, 'test/Mask/')
        else:
            self.Xgt_path = os.path.join(self.dir, 'val/Target/')
            self.Xma_path = os.path.join(self.dir, 'val/Baseline/')
            self.XLI_path = os.path.join(self.dir, 'val/LI/')
            self.Mask_path = os.path.join(self.dir, 'val/Mask/')

        # List all raw files in the Xgt_path directory
        self.mat_files = sorted([f for f in os.listdir(self.Xgt_path) if f.endswith('.raw')])
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        gt_filename = self.mat_files[idx % self.file_num].strip()
        base_filename = gt_filename.replace('nometal', 'metalart').replace('Target', 'Baseline')

        Xgt_absdir = os.path.join(self.Xgt_path, gt_filename)
        Xma_absdir = os.path.join(self.Xma_path, base_filename)
        XLI_absdir = os.path.join(self.XLI_path, base_filename.replace('img', 'sino'))
        Mask_absdir = os.path.join(self.Mask_path, base_filename.replace('img', 'mask').replace("metalart_mask","metalonlymask_img"))

        # Read the images using rawread function
        Xgt = rawread(Xgt_absdir, [512, 512], 'float')
        Xma = rawread(Xma_absdir, [512, 512], 'float')
        XLI = rawread(XLI_absdir, [512, 512], 'float')
        M512 = rawread(Mask_absdir, [512, 512], 'float')

        Xgt = (Xgt +1500)/5000    # FUXIN
        Xma = (Xma +1500)/5000    # FUXIN
        XLI = (XLI +1500)/5000    # FUXIN

        # Example usage:
        check_nan_inf(Xgt, "Xgt", Xgt_absdir)
        check_nan_inf(Xma, "Xma", Xma_absdir)
        check_nan_inf(XLI, "XLI", XLI_absdir)
        check_nan_inf(M512, "M512", Mask_absdir)

        # Resize the images
        Xgt_resized = np.array(Image.fromarray(Xgt).resize(self.output_size, PIL.Image.BILINEAR))
        Xma_resized = np.array(Image.fromarray(Xma).resize(self.output_size, PIL.Image.BILINEAR))
        XLI_resized = np.array(Image.fromarray(XLI).resize(self.output_size, PIL.Image.BILINEAR))

        # Resize the mask
        M = np.array(Image.fromarray(M512).resize(self.output_size, PIL.Image.BILINEAR))

        # Normalize and clip images
        Xgtclip = np.clip(Xgt_resized, 0, 1)
        Xmaclip = np.clip(Xma_resized, 0, 1)
        XLIclip = np.clip(XLI_resized, 0, 1)
        M = np.clip(M, 0, 1)

        O = Xmaclip * 255.0
        O, row, col = self.crop(O)
        B = Xgtclip * 255.0
        B = B[row: row + self.patch_size, col: col + self.patch_size]
        LI = XLIclip * 255.0
        LI = LI[row: row + self.patch_size, col: col + self.patch_size]
        M = M[row: row + self.patch_size, col: col + self.patch_size]

        # Convert to float32
        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)

        # Apply augmentations if needed
        if self.augment:
            O, B, LI, Mask = augment(O, B, LI, Mask)

        # Prepare tensors
        O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
        B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
        LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))

        non_Mask = 1 - Mask  # non-metal region

        return gt_filename, torch.from_numpy(O.copy()), torch.from_numpy(B.copy()), torch.from_numpy(LI.copy()), torch.from_numpy(non_Mask.copy())  # O: Baseline, B: Target, LI: LI, non_Mask: non-metal region
        # return torch.from_numpy(O.copy()), torch.from_numpy(B.copy()), torch.from_numpy(LI.copy()), torch.from_numpy(non_Mask.copy())


    def crop(self, img):
        h, w = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if h == p_h:
            r = 0
            c = 0
            O = img
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
            O = img[r: r + p_h, c: c + p_w]
        return O, r, c
