import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)    # choose resnet18;34;50;101;152
# resnet_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)    # resnet50 v1.5
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

submodel_layer1 = nn.Sequential(*list(resnet_model.children())[:5])    # 根据resnet_model的结构，取其前6层（此时对应layer2）
submodel_layer2 = nn.Sequential(*list(resnet_model.children())[:6])    # （此时对应layer3）
submodel_layer3 = nn.Sequential(*list(resnet_model.children())[:7])
submodel_layer4 = nn.Sequential(*list(resnet_model.children())[:8])

resnet_model.eval().to(device)
submodel_layer1.eval().to(device)
submodel_layer2.eval().to(device)
submodel_layer3.eval().to(device)
submodel_layer4.eval().to(device)

uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)    # prepare images 4*3*244*244

with torch.no_grad():
    output = torch.nn.functional.softmax(resnet_model(batch), dim=1)
    output_layer1 = submodel_layer1(batch)    # m*n*56*56  m=4,n=256
    output_layer2 = submodel_layer2(batch)    # m*n*28*28  m=4,n=512
    output_layer3 = submodel_layer3(batch)    # m*n*14*14  m=4,n=1024
    output_layer4 = submodel_layer4(batch)    # m*n*7*7    m=4,n=2048
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