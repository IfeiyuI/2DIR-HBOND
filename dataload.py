# from torch.utils.data import Dataset
# import numpy as np
# import os
# import cv2
# import torch
# import pandas as pd
# from torchvision import transforms
#
# class Mydata(Dataset):
#     def __init__(self, dataset_dir, images_dir, label_dir, train_data=True, print_id=False, source_img=False):
#         super(Mydata).__init__()
#         self.train_data = train_data
#         self.print_id = print_id
#         self.image_dir = os.path.join(dataset_dir, images_dir)
#         self.im_names = sorted(os.listdir(self.image_dir))
#
#         self.label_dir = os.path.join(dataset_dir, label_dir, 'label_h_sasa.xlsx')
#         self.files = pd.read_excel(self.label_dir)
#
#         self.files['id'] = self.files['id'].astype(str)
#         self.files = self.files.sort_values(by='id')
#
#         self.labels = self.files[["num_h", "SASA_Total", "SASA_Hydrophobic", "SASA_Hydrophilic"]].values
#
#         self.label_id = self.files['id'].tolist()  # 用于验证数据是否一一对应
#         self.data_id = self.label_id if (self.print_id) else None
#         self.source_img = source_img
#
#         self.train_transformer = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
#             transforms.RandomRotation(10),  # 旋转
#         ])
#
#         self.test_transformer = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
#         ])
#
#         self.not_transformer = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#     def __getitem__(self, index):
#         im_name = self.im_names[index]
#         im_id = int(''.join(filter(str.isdigit, im_name)))
#         im_path = os.path.join(self.image_dir, im_name)
#         im = cv2.imread(im_path, cv2.IMREAD_COLOR)
#
#         if self.source_img:
#             im_tensor = self.not_transformer(im)
#
#         if self.train_data:
#             im = self.train_transformer(im)
#         else:
#             im = self.test_transformer(im)
#
#         label = torch.tensor(self.labels[index], dtype=torch.float32)
#
#         assert (im_name.split('.')[0] == self.label_id[index]), f"Image ID {im_name} and label ID {self.label_id[index]} mismatch!"
#
#         if self.print_id:
#             if self.source_img:
#                 return im, label, im_tensor, im_id
#             else:
#                 return im, label, im_id
#         else:
#             return im, label
#
#     def __len__(self):
#         return len(self.im_names)
#
#
# # from torch.utils.data import Dataset
# # import numpy as np
# # import os
# # import torch
# # import pandas as pd
# # from torchvision import transforms
# #
# # class Mydata(Dataset):
# #     def __init__(self, dataset_dir, images_dir, label_dir, train_data=True, print_id=False, source_img=False):
# #         super(Mydata).__init__()
# #         self.train_data = train_data
# #         self.print_id = print_id
# #         self.image_dir = os.path.join(dataset_dir, images_dir)
# #         self.label_file = os.path.join(dataset_dir, label_dir, 'label_h_sasa.xlsx')
# #
# #         # 从 Excel 文件中加载标签
# #         self.labels_df = pd.read_excel(self.label_file)
# #
# #         # 从 npz 文件加载图像数据
# #         self.image_files = {int(file.split('.')[0]): np.load(os.path.join(self.image_dir, file))['arr_0']
# #                             for file in os.listdir(self.image_dir) if file.endswith('.npz')}
# #
# #         # 匹配标签与图像
# #         self.labels = []
# #         self.images = []
# #         self.ids = []  # 存储 npz 文件的数字部分，即 img_id
# #         for idx, row in self.labels_df.iterrows():
# #             img_id = row['id']
# #             if img_id in self.image_files:
# #                 self.labels.append(row[["num_h", "SASA_Total", "SASA_Hydrophobic", "SASA_Hydrophilic"]].values)
# #                 self.images.append(self.image_files[img_id])
# #                 self.ids.append(img_id)  # 保存 img_id 以供打印
# #
# #         self.transform = transforms.Compose([
# #             transforms.ToTensor(),  # 转换为张量
# #             transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True))  # 将图像压缩到1通道
# #         ])
# #
# #     def __len__(self):
# #         return len(self.labels)
# #
# #     def __getitem__(self, idx):
# #         image = self.images[idx]
# #         label = self.labels[idx]
# #         img_id = self.ids[idx]  # 获取当前图像的 ID
# #
# #         if self.transform:
# #             image = self.transform(image)
# #
# #         # 打印当前图像的 ID 和标签的 ID 信息，仅在 print_id 为 True 时打印
# #         if self.print_id:
# #             print(f"Loading npz ID: {img_id}, Label ID: {self.labels_df.at[idx, 'id']}")
# #
# #         return image, torch.tensor(label, dtype=torch.float32)  # 转换标签为张量
#
#





#
# # 未归一化
# import os
# import torch
# import pandas as pd
# import numpy as np
# import cv2
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image  # 添加 PIL 库
#
# class Mydata(Dataset):
#     def __init__(self, dataset_dir, images_dir, label_dir, return_ids=False, train_data=True, source_img=False):
#         super(Mydata).__init__()
#         self.train_data = train_data
#         self.return_ids = return_ids  # 更新为 return_ids
#         self.image_dir = os.path.join(dataset_dir, images_dir)
#         self.im_names = sorted(os.listdir(self.image_dir))
#
#         self.label_dir = os.path.join(dataset_dir, label_dir, 'label_h_sasa.xlsx')
#         self.files = pd.read_excel(self.label_dir)
#
#         self.files['id'] = self.files['id'].astype(str)
#         self.files = self.files.sort_values(by='id')
#
#         self.labels = self.files[["num_h", "SASA_Hydrophobic", "Length", "pI"]].values #  "SASA_Total",, "SASA_Hydrophilic", "Relative Hydrophobic area", "Relative Hydrophilic area", "SASA_Total", "SASA_Hydrophobic", "SASA_Hydrophilic", "R_Ho", "R_Hi", "GC", "Length", "pI"
#
#
#         self.label_id = self.files['id'].tolist()  # 用于验证数据是否一一对应
#         self.data_id = self.label_id if self.return_ids else None
#         self.source_img = source_img
#
#         self.train_transformer = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.RandomRotation(10),  # 旋转
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
#         ])
#
#         self.test_transformer = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
#         ])
#
#         self.not_transformer = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#     def __getitem__(self, index):
#         im_name = self.im_names[index]
#         im_id = int(''.join(filter(str.isdigit, im_name.split('_')[0])))  # 确保 im_id 是整数
#         im_path = os.path.join(self.image_dir, im_name)
#         im = cv2.imread(im_path, cv2.IMREAD_COLOR)
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 转换为 RGB
#
#         im = Image.fromarray(im)  # 将 numpy 数组转换为 PIL.Image
#
#         if self.source_img:
#             im_tensor = self.not_transformer(im)
#
#         transformer = self.train_transformer if self.train_data else self.test_transformer
#         im = transformer(im)
#
#         # 添加调试信息，打印图像尺寸
#         # print(f"Image {im_name} size after transform: {im.shape}")
#
#         label = torch.tensor(self.labels[index], dtype=torch.float32)
#
#         assert (im_name.split('.')[0] == self.label_id[index]), f"Image ID {im_name} and label ID {self.label_id[index]} mismatch!"
#
#         if self.return_ids:
#             if self.source_img:
#                 return im, label, im_tensor, im_id
#             else:
#                 return im, label, im_id
#         else:
#             return im, label
#
#     def __len__(self):
#         return len(self.im_names)



# 有归一化
import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # 添加 PIL 库

class Mydata(Dataset):
    def __init__(self, dataset_dir, images_dir, label_dir, return_ids=False, train_data=True, source_img=False):
        super(Mydata).__init__()
        self.train_data = train_data
        self.return_ids = return_ids  # 更新为 return_ids
        self.image_dir = os.path.join(dataset_dir, images_dir)
        self.im_names = sorted(os.listdir(self.image_dir))

        self.label_dir = os.path.join(dataset_dir, label_dir, 'label_h_sasa.xlsx')
        self.files = pd.read_excel(self.label_dir)

        self.files['id'] = self.files['id'].astype(str)
        self.files = self.files.sort_values(by='id')

        self.labels = self.files[["num_h", "SASA_Hydrophobic", "Length", "pI"]].values

        # 计算每个标签列的最小值和最大值，用于归一化
        self.labels_min = np.min(self.labels, axis=0)
        self.labels_max = np.max(self.labels, axis=0)

        # 将标签归一化到 [0, 1] 范围
        self.labels = (self.labels - self.labels_min) / (self.labels_max - self.labels_min)

        self.label_id = self.files['id'].tolist()  # 用于验证数据是否一一对应
        self.data_id = self.label_id if self.return_ids else None
        self.source_img = source_img

        self.train_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomRotation(10),  # 旋转
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像标准化
        ])

        self.test_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像标准化
        ])

        self.not_transformer = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_id = int(''.join(filter(str.isdigit, im_name.split('_')[0])))  # 确保 im_id 是整数
        im_path = os.path.join(self.image_dir, im_name)
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        im = Image.fromarray(im)  # 将 numpy 数组转换为 PIL.Image

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
        """将归一化后的标签反归一化为原始尺度"""
        return normalized_labels * (self.labels_max - self.labels_min) + self.labels_min
