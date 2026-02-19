import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LoveDADataset(Dataset):
    def __init__(self, root_dir, split='train', crop_size=(512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.root_dir = root_dir
        self.split = split
        self.debug = False  
        expected_dirs = ['Train', 'Val']
        if not all(os.path.exists(os.path.join(root_dir, d)) for d in expected_dirs):
            raise ValueError(f"Invalid LoveDA root directory: {root_dir}. Expected subdirs: {expected_dirs}")
        split_map = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
        if split not in split_map:
            raise ValueError(f"Unknown split: {split}")
        actual_split = split_map[split]
        self.image_files = []
        self.label_files = []
        for subset in ['Rural', 'Urban']:
            image_path = os.path.join(root_dir, actual_split, subset, 'images_png')
            label_path = os.path.join(root_dir, actual_split, subset, 'masks_png')
            if os.path.exists(image_path) and os.path.exists(label_path):
                img_names = sorted([f for f in os.listdir(image_path) if f.endswith('.png') and not f.endswith(':Zone.Identifier')])
                lbl_names = sorted([f for f in os.listdir(label_path) if f.endswith('.png') and not f.endswith(':Zone.Identifier')])
                common_names = set(img_names) & set(lbl_names)
                for name in sorted(common_names):
                    self.image_files.append(os.path.join(image_path, name))
                    self.label_files.append(os.path.join(label_path, name))
            else:
                print(f"[WARN] Skipping subset {subset} for split {actual_split}: missing images_png or masks_png")
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {root_dir} for split={split}")
        print(f"[INFO] Loaded {len(self.image_files)} images for split={split}")
        self.transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize(crop_size, interpolation=Image.NEAREST),
            transforms.CenterCrop(crop_size),
        ])

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        if self.debug:
            print(f"[DEBUG] Loading image: {image_path}")
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = Image.fromarray(np.array(image), mode='RGB')
        except Exception as e:
            print(f"⚠️ Corrupted image: {image_path}, error: {e}")
            return self.__getitem__((index + 1) % len(self))
        if self.debug:
            print(f"[DEBUG] Loading label: {label_path}")
        try:
            label = Image.open(label_path)
            if label.mode == 'I':
                label = Image.fromarray(np.array(label, dtype=np.uint8), mode='L')
            elif label.mode != 'L':
                label = label.convert('L')
        except Exception as e:
            print(f"⚠️ Corrupted label: {label_path}, error: {e}")
            return self.__getitem__((index + 1) % len(self))
        if self.split == 'train':
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            label = self.label_transform(label)
        else:
            image = self.transform(image)
            label = self.label_transform(label)
        label_np = np.array(label, dtype=np.int64)
        invalid_mask = (label_np < 0) | (label_np > 6)
        if invalid_mask.any():
            label_np[invalid_mask] = 0
        label_tensor = torch.from_numpy(label_np)
        return image, label_tensor

    def __len__(self):
        return len(self.image_files)

def get_dataset(config):
    return LoveDADataset(
        root_dir=config['data_root'],
        split=config['split'],
        crop_size=config['crop_size'],
        mean=config['mean'],
        std=config['std']
    )