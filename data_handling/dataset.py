import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset

from data_handling.preprocessing import *


class CityscapesDataset(Dataset):
    def __init__(self, input_dataframe: pd.DataFrame, root_dir: str, KeysOfInterest: List[str], data_transform:Compose):
        self.root_dir = root_dir
        self.koi = KeysOfInterest
        self.input_dataframe = input_dataframe[self.koi]
        self.data_transform=data_transform

    def __getitem__(self, item: int):
        sample = self.input_dataframe.iloc[item]

        sample_dict = self.data_transform(sample)

        image = sample_dict['image']
        mask = sample_dict['mask']
        return image, mask
        
    def __len__(self):
        return len(self.input_dataframe)
    
class DatasetHandler:
    def __init__(self, input_dataframe: pd.DataFrame, batch_size=4, num_workers=0,prefetch_factor=0, shuffle=True):
        test_data_transform=Compose([
            LoadImage( keys= ['image_path'] ),
            LoadMask( keys= ['mask_path'] ),
            RemapTrainIds( keys= ['mask'] ),
            TensorImage(keys=['image']),
            TensorMask(keys = ['mask'])
        ])
        self.dataset =CityscapesDataset(input_dataframe=input_dataframe,
                root_dir="",
                KeysOfInterest=["image_path","mask_path"],
                data_transform=test_data_transform)
        
        if num_workers == 0:
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        else:
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, shuffle=shuffle)
        
        
    def save_samples_in_batch(self, num_samples=4):
        """
        Show a few samples from the DataLoader.
        
        Args:
            num_samples (int): Number of samples to display.
        """
        # Get one batch of data from the DataLoader
        data_iter = iter(self.dataloader)
        images, masks = next(data_iter)
        
        # Convert tensors to numpy arrays and denormalize if needed
        images = images.numpy().transpose(0, 2, 3, 1) 

        masks = masks.numpy() 

        # Number of samples to show
        num_samples = min(num_samples, len(images))

        # Create a figure to plot
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
        fig.suptitle('Sample Images and Masks')

        for i in range(num_samples):
            # Plot image
            axes[i, 0].imshow(images[i], cmap='gray')
            axes[i, 0].set_title(f'Image {i+1}')
            axes[i, 0].axis('off')

            # Plot mask
            axes[i, 1].imshow(masks[i], cmap='gray')
            axes[i, 1].set_title(f'Car Mask {i+1}')
            axes[i, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("./results/sample_images_and_masks.png")
    
    def get_dataset(self):
        return self.dataset
    
    def get_dataloader(self):
        return self.dataloader
    
