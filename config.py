import torch
import numpy as np

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

dataset_dir_train = r"./data/train"
dataset_dir_test = r"./data/test"
# dataset_dir_train = r"./data/base_all"
# dataset_dir_test = r"./data/base_all"
images_dir = r"images"
label_dir = r"label"

# images_base_dir = r"images_base_4086"
# label_base_dir = r"label_base_4086"
images_base_dir = r"images_base_more"
label_base_dir = r"label_base_more"

batch_size = 8
learn_rate = 0.0001
epochs = 80

# pearson相关系数， 通过torch计算，就不需要将数据转移到cpu上进行
def pearson(a, b):
    a_mean = torch.mean(a)
    b_mean = torch.mean(b)
    numerator = torch.sum((a - a_mean) * (b - b_mean))
    denominator_a = torch.sqrt(torch.sum((a - a_mean)**2))
    denominator_b = torch.sqrt(torch.sum((b - b_mean)**2))
    pearson_ = numerator / (denominator_a)  / denominator_b
    return pearson_

def pearson_np(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    numerator = np.sum((a - a_mean) * (b - b_mean))
    denominator_a = np.sqrt(np.sum((a - a_mean)**2))
    denominator_b = np.sqrt(np.sum((b - b_mean)**2))
    pearson_ = numerator / (denominator_a)  / denominator_b
    return pearson_
