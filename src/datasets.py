import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]






from PIL import Image
from torchvision import transforms


class ThingsMEGDataset_2(torch.utils.data.Dataset):  # load images too
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()

        #self.transform = transforms.Compose([
        #    transforms.Resize((224, 224)),  # Resize images to a fixed size
        #    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
        #])
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            self.images = torch.load(os.path.join(data_dir, f"{split}_images.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        #if split in ["train", "val"]:
        #  completed_paths = []
        #
        #  with open(f'data/{split}_image_paths.txt', 'r') as f:
        #    self.image_paths = f.read().splitlines()
        #  
        #  for path in self.image_paths:
        #    if not "/" in path:  # フォルダ名が欠落していたら, ファイル名から推測して補完
        #      #foldername = path.split("_")[0]
        #      foldername = '_'.join(path.split("_")[:-1])
        #      path = os.path.join(foldername+"/", path)
        #    path = os.path.join("data/Images/", path)
        #    completed_paths.append(path)
        #  
        #  self.image_paths = completed_paths


        ## 画像のプリロード
        #self.images = []
        #for path in self.image_paths:
        #    image = Image.open(path).convert("RGB")
        #    image = self.transform(image)
        #    self.images.append(image)
        
        ## save as tensor
        #tensor_path = os.path.join(data_dir, f"{split}_images.pt")
        #torch.save(self.images, tensor_path)
          

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            #image = Image.open(self.image_paths[i]).convert("RGB")
            #image = self.transform(image)  # Transform the image to a tensor
            return self.X[i], self.y[i], self.images[i], self.subject_idxs[i]  #, self.image_paths[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]