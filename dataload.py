import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Mydata(Dataset):
    def __init__(self, dataset_dir, images_dir, label_dir, return_ids=False, train_data=True, source_img=False):
        super(Mydata, self).__init__()
        self.train_data = train_data
        self.return_ids = return_ids
        self.source_img = source_img

        self.image_dir = os.path.join(dataset_dir, images_dir)
        self.im_names = sorted(os.listdir(self.image_dir))

        label_path = os.path.join(dataset_dir, label_dir, 'label_h_sasa.xlsx')
        df = pd.read_excel(label_path)
        df['id'] = df['id'].astype(str)
        df = df.sort_values(by='id')

        self.labels = df[["num_h", "SASA_Hydrophobic", "Length", "pI"]].values
        self.labels_min = np.min(self.labels, axis=0)
        self.labels_max = np.max(self.labels, axis=0)
        self.labels = (self.labels - self.labels_min) / (self.labels_max - self.labels_min)

        self.label_id = df['id'].tolist()
        self.data_id = self.label_id if self.return_ids else None

        self.train_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.not_transformer = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_id = int(''.join(filter(str.isdigit, im_name.split('_')[0])))
        im_path = os.path.join(self.image_dir, im_name)
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        if self.source_img:
            im_tensor = self.not_transformer(im)

        transformer = self.train_transformer if self.train_data else self.test_transformer
        im = transformer(im)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        assert (im_name.split('.')[0] == self.label_id[index]), f"Image ID {im_name} and label ID {self.label_id[index]} mismatch!"

        if self.return_ids:
            if self.source_img:
                return im, label, im_tensor, im_id
            else:
                return im, label, im_id
        else:
            return im, label

    def __len__(self):
        return len(self.im_names)

    def inverse_normalize_labels(self, normalized_labels):
        """Convert normalized labels back to original scale."""
        return normalized_labels * (self.labels_max - self.labels_min) + self.labels_min
