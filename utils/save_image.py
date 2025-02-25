import tifffile
import os
import matplotlib.pyplot as plt
import numpy as np

def normalize(data):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)
    
def window_normalize(data, window_min=-700, window_max=1000):
    # Clip the data to the specified window
    data_clipped = np.clip(data, window_min, window_max)
    # Normalize the clipped data to [0, 1]
    data_min, data_max = data_clipped.min(), data_clipped.max()
    return (data_clipped - data_min) / (data_max - data_min)

def window_clip(data, window_min=-700, window_max=1000):
    # Clamp intensity values within the specified window
    data[data > window_max] = window_max
    data[data < window_min] = window_min
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)

def imwrite(idx, dir, datalist, format='tiff', bit_depth=16, gamma=1.0, window_norm=False, window_min=-700, window_max=1000):
    for i in range(len(datalist)):
        file_dir = os.path.join(dir[i], f"{str(idx)}.{format}")
        image_data = datalist[i].data.cpu().numpy().squeeze().astype(np.float32)
        

        if window_norm == True:
            image_data = window_normalize(image_data, window_min=window_min, window_max=window_max)
        else:
            # Normalize and apply gamma correction
            image_data = normalize(image_data)
        
        if format == 'tiff':
            if bit_depth == 8:
                image_data = (image_data * 255).astype(np.uint8)
            elif bit_depth == 16:
                image_data = (image_data * 65535).astype(np.uint16)
            elif bit_depth == 32:
                # Keep data as float32
                pass
            else:
                raise ValueError("Unsupported bit depth for TIFF format")
            tifffile.imsave(file_dir, image_data)
        elif format == 'png':
            # Ensure the data is in the range [0, 1] before saving as PNG
            plt.imsave(file_dir, image_data, cmap="gray")

def masked_imwrite(idx, dir, datalist, format='tiff', bit_depth=16, gamma=1.0, window_norm=False, window_min=-700, window_max=1000, mask=None):
    mask = mask.data.cpu().numpy().squeeze().astype(np.bool8)
    for i in range(len(datalist)):
        file_dir = os.path.join(dir[i], f"{str(idx)}.{format}")
        image_data = datalist[i].data.cpu().numpy().squeeze().astype(np.float32)
        
        

        if window_norm == True:
            image_data = window_normalize(image_data, window_min=window_min, window_max=window_max)
        else:
            # Normalize and apply gamma correction
            image_data = normalize(image_data)
        if format == 'tiff':
            # Apply the mask to the image data
            image_data = np.where(~mask, 1.0, image_data)  # Set unmasked parts to 1.0

            if bit_depth == 8:
                image_data = (image_data * 255).astype(np.uint8)
            elif bit_depth == 16:
                image_data = (image_data * 65535).astype(np.uint16)
            elif bit_depth == 32:
                # Keep data as float32
                pass
            else:
                raise ValueError("Unsupported bit depth for TIFF format")
            tifffile.imsave(file_dir, image_data)
        elif format == 'png':
            # Ensure the data is in the range [0, 1] before saving as PNG
            plt.imsave(file_dir, np.where(~mask, 1.0, image_data), cmap="gray")


def marked_masked_imwrite(idx, dir, datalist, format='tiff', bit_depth=16, gamma=1.0, window_norm=False, window_min=-700, window_max=1000, mask=None):
    mask = mask.data.cpu().numpy().squeeze().astype(np.bool8)
    for i in range(len(datalist)):
        file_dir = os.path.join(dir[i], f"{str(idx)}.{format}")
        image_data = datalist[i].data.cpu().numpy().squeeze().astype(np.float32)

        if window_norm:
            image_data = window_normalize(image_data, window_min=window_min, window_max=window_max)
        else:
            # Normalize and apply gamma correction
            image_data = normalize(image_data)

        # Scale image data based on bit depth
        if bit_depth == 8:
            image_data = (image_data * 255).astype(np.uint8)
        elif bit_depth == 16:
            image_data = (image_data * 65535).astype(np.uint16)
        elif bit_depth != 32:
            raise ValueError("Unsupported bit depth for TIFF format")

        # Create RGB color image, keeping masked areas grayscale and making non-masked areas red
        color_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.float32)
        color_image[..., 0] = np.where(~mask, 1.0, image_data)  # Red channel: red for non-masked, grayscale for masked
        color_image[..., 1] = image_data * mask  # Green channel: grayscale for masked area
        color_image[..., 2] = image_data * mask  # Blue channel: grayscale for masked area

        # Normalize to 0..1 range for saving as PNG
        color_image = np.clip(color_image, 0, 1)

        if format == 'tiff':
            tifffile.imsave(file_dir, (color_image * 65535).astype(np.uint16) if bit_depth == 16 else (color_image * 255).astype(np.uint8))
        elif format == 'png':
            plt.imsave(file_dir, color_image)
