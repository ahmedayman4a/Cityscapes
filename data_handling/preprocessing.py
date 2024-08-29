import torch
import cv2
import numpy as np

class LoadImage:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                image = cv2.imread(sample[key], cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
                sample["image"] =image
            else:
                raise KeyError(f"Key '{key}' not found in the sample")
        
        return sample
    
class LoadMask:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                image = cv2.imread(sample[key], cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
                sample["mask"] =image
            else:
                raise KeyError(f"Key '{key}' not found in the sample")
        
        return sample


class RemapTrainIds:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                image = sample[key]
                image[image == 19] = 255
                image[image == 0] = 19
                image[image == 255] = 0
                sample[key] = image
            else:
                raise KeyError(f"Key '{key}' not found in the sample")
        
        return sample
    
    

class TensorImage:
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, sample):
        processed_images = {}
        for key in self.keys:
            if key in sample:
                processed_images[key] = torch.from_numpy(np.transpose(sample[key], (2, 0, 1)) / 255.0).float()
            else:
                raise KeyError(f"Key '{key}' not found in the sample")
        
        sample.update(processed_images)
        
        return sample

class TensorMask:
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, sample):
        processed_masks = {}
        for key in self.keys:
            if key in sample:
                processed_masks[key] = torch.from_numpy(sample[key]).long()
            else:
                raise KeyError(f"Key '{key}' not found in the sample")
        
        sample.update(processed_masks)
        
        return sample
