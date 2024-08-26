import utils
import torch
import os
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt

replay_buffer = torch.load('saved_buffers/replay_buffer_500000.pt')
test_buffer_1 = torch.load('saved_buffers/test_buffer_1_500000.pt')
test_buffer_2 = torch.load('saved_buffers/test_buffer_2_500000.pt')
test_buffer_3 = torch.load('saved_buffers/test_buffer_3_500000.pt')

plt.imshow(np.transpose(test_buffer_3._obses[00000][1].frames[0],(1,2,0)))
plt.savefig('test_buffer_3_frame_0.png')

# TODO: T-SNE or UMDP

# TODO: ResNet50 for encoder