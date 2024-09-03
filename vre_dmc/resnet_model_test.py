import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
import os
import sys
sys.path.append(os.getcwd())
from src.arguments import parse_args

args = parse_args()
de_num = args.de_num
warnings.filterwarnings('ignore')

device = torch.device(f"cuda:{de_num}") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

class resnet_model:
    def __init__(self, model_name):
        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)    # choose resnet18;34;50;101;152
        # resnet_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)    # resnet50 v1.5
        self.submodel_layer1 = nn.Sequential(*list(self.resnet_model.children())[:5])    # 根据resnet_model的结构，取其前6层（此时对应layer2）
        self.submodel_layer2 = nn.Sequential(*list(self.resnet_model.children())[:6])    # （此时对应layer3）
        self.submodel_layer3 = nn.Sequential(*list(self.resnet_model.children())[:7])
        self.submodel_layer4 = nn.Sequential(*list(self.resnet_model.children())[:8])

        self.resnet_model.eval().to(device)
        self.submodel_layer1.eval().to(device)
        self.submodel_layer2.eval().to(device)
        self.submodel_layer3.eval().to(device)
        self.submodel_layer4.eval().to(device)


if __name__ == '__main__':
    RS = resnet_model('resnet50')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
    uris = [
        'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
    ]

    batch = torch.cat(
        [utils.prepare_input_from_uri(uri) for uri in uris]
    ).to(device)    # 从urls中读取图片，格式为n*3*244*244的array

    with torch.no_grad():
        output = torch.nn.functional.softmax(RS.resnet_model(batch), dim=1)
        output_layer1 = RS.submodel_layer1(batch)    # m*n*56*56  m=4,n=256
        output_layer2 = RS.submodel_layer2(batch)    # m*n*28*28  m=4,n=512
        output_layer3 = RS.submodel_layer3(batch)    # m*n*14*14  m=4,n=1024
        output_layer4 = RS.submodel_layer4(batch)    # m*n*7*7    m=4,n=2048
        # 第四层通过average pooling，得到m*n*1*1的向量    m=4,n=2048

    results = utils.pick_n_best(predictions=output, n=5)

    for uri, result in zip(uris, results):
        img = Image.open(requests.get(uri, stream=True).raw)
        img.thumbnail((256,256), Image.LANCZOS)    # resize image
        plt.imshow(img)
        plt.imsave('test.jpg', img)
        # plt.imsave('test.jpg', np.sum(np.array(output_layer1[0].cpu()),axis=0))    # 存储查看layer1的输出图片（将通道求和了）
        plt.show()
        print(result)