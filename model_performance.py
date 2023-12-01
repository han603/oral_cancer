import torch
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 模型的类
from model.capsnet import *
#
# 模型路径
save_path = r'D:\pythonProject\oral_cancer\模型输出\tl2\capsnet\best\capsnet-oral_epoch43.pth'

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load(save_path, map_location=device)


# 数据处理
normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize
])

# 读取图像数据
batch_size = 32
train_dataset = ImageFolder('D:\pythonProject\oral_cancer\Dataset\Oral Cancer5/train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder('D:\pythonProject\oral_cancer\Dataset\Oral Cancer5/test/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('{0} for train. {1} for val'.format(len(train_dataset), len(test_dataset)))


def model_performence(dataloader):
    '''
    传入dataloader文件
    返回: 预测标签， 真实标签， 类型为numpy的array
    '''
    net.eval()
    y_hat_label = []
    y_true_label = []
    for (X_batch, y_batch) in tqdm(dataloader, leave=False):

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_hat = net(X_batch)
        y_hat = y_hat.norm(dim=-1)
        _, y_hat = torch.max(y_hat, 1)
        y_hat_label.extend(y_hat.tolist())
        y_true_label.extend(y_batch.tolist())
    return y_hat_label, y_true_label

y_hat_label, y_true_label = model_performence(train_loader)
print('train dataset:')
print('accuracy', round(accuracy_score(y_true_label, y_hat_label), 4))
print('recall', round(recall_score(y_true_label, y_hat_label), 4))
print('precision', round(precision_score(y_true_label, y_hat_label), 4))
print('f1_score', round(f1_score(y_true_label, y_hat_label), 4))

y_hat_label, y_true_label = model_performence(test_loader)
print('test dataset:')
print('accuracy', round(accuracy_score(y_true_label, y_hat_label), 4))
print('recall', round(recall_score(y_true_label, y_hat_label), 4))
print('precision', round(precision_score(y_true_label, y_hat_label), 4))
print('f1_score', round(f1_score(y_true_label, y_hat_label), 4))


