import utils
import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
from augmentations import random_conv, random_overlay, random_shift, random_choose_double, random_choose
from umap import UMAP
from resnet_model_test import resnet_model
from arguments import parse_args
import algorithms.modules as m

# 分开降维_随机初始化的cnn

def main():
    args = parse_args()
    de_num = args.de_num
    shared_cnn = m.SharedCNN((3,84,84), args.num_shared_layers, args.num_filters).cuda(de_num)    # shared_11,filters_32
    head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda(de_num)
    encoder = m.Encoder(
        shared_cnn,
        head_cnn,
        m.RLProjection(head_cnn.out_shape, args.projection_dim)
    ).to(de_num)
    replay_buffer = torch.load('saved_buffers_sac/replay_buffer_500000.pt')
    test_buffer_1 = torch.load('saved_buffers_sac/test_buffer_1_500000.pt')
    test_buffer_2 = torch.load('saved_buffers_sac/test_buffer_2_500000.pt')
    test_buffer_3 = torch.load('saved_buffers_sac/test_buffer_3_500000.pt')
    # RL更新所用的frame是9*84*84，一次采了三张图片
    batch_size = 200
    obses_rb = buffer_sample_obs(replay_buffer, batch_size)
    obses_tb1 = buffer_sample_obs(test_buffer_1, batch_size)
    obses_tb2 = buffer_sample_obs(test_buffer_2, batch_size)
    obses_tb3 = buffer_sample_obs(test_buffer_3, batch_size)

    # np.moveaxis(array,0,-1)    # 将array的第0维移动维最后的维度
    rb_tensor = torch.cat([torch.tensor([(np.array(obses_rb[i][0]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)
                          ,torch.tensor([(np.array(obses_rb[i][1]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)],dim=0).float().to(f'cuda:{de_num}')
    tb1_tensor = torch.cat([torch.tensor([(np.array(obses_tb1[i][0]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)
                          ,torch.tensor([(np.array(obses_tb1[i][1]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)],dim=0).float().to(f'cuda:{de_num}')
    tb2_tensor = torch.cat([torch.tensor([(np.array(obses_tb2[i][0]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)
                          ,torch.tensor([(np.array(obses_tb2[i][1]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)],dim=0).float().to(f'cuda:{de_num}')
    tb3_tensor = torch.cat([torch.tensor([(np.array(obses_tb3[i][0]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)
                          ,torch.tensor([(np.array(obses_tb3[i][1]).reshape(3,3,84,84)) for i in range(batch_size)]).reshape(-1,3,84,84)],dim=0).float().to(f'cuda:{de_num}')

    # ---------use augmentations ---------
    with torch.no_grad():
        rb_shift = random_shift(rb_tensor)
        rb_conv = random_conv(rb_shift)
        rb_over = random_overlay(rb_shift)
        rb_cho_do = random_choose_double(rb_shift)
        rb_cho = random_choose(rb_shift)

    # ---------resnet&umap----------------
    # RS = resnet_model('resnet50')
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, densmap = True, dens_lambda = 2)

    with torch.no_grad():
        out_rb = encoder(rb_tensor)
        out_shift = encoder(rb_shift)
        out_conv = encoder(rb_conv)
        out_over = encoder(rb_over)
        out_cho_double = encoder(rb_cho_do)
        out_cho = encoder(rb_cho)
        out_tb1 = encoder(tb1_tensor)
        out_tb2 = encoder(tb2_tensor)
        out_tb3 = encoder(tb3_tensor)
    
        length = out_rb.shape[0]
        # ----------------sample&cat--------------------------------
        # out_shift = torch.cat([out_rb[random_idx(length),:],out_shift[random_idx(length),:]])
        # out_conv = torch.cat([out_shift[random_idx(length),:],out_conv[random_idx(length),:]])
        # out_over = torch.cat([out_shift[random_idx(length),:],out_over[random_idx(length),:]])
        # out_cho_double = torch.cat([out_shift[random_idx(length),:],out_cho_double[random_idx(length),:]])
        # out_cho = torch.cat([out_shift[random_idx(length),:],out_cho[random_idx(length),:]])

    # aug_mode = ["train_shift_conv", "train_shift_over", "train_shift_cho_double","train_shift_cho"]
    aug_mode = ["train_ori","train_shift", "train_conv", "train_over", "train_cho_double","train_cho"]
    test_mode = ["color_hard","video_easy","video_hard"]
    # out_train = [out_conv, out_over, out_cho_double, out_cho]
    out_train = [out_rb, out_shift, out_conv, out_over, out_cho_double, out_cho]
    out_test = torch.cat([out_tb1, out_tb2, out_tb3],dim=0)

    palette = sns.color_palette("Set1",10)
    sns.set_palette(palette)
    # palette=['#DC565F', '#3E61AC']

    for train,i in zip(out_train,range(len(out_train))):
        for j in range(len(test_mode)):
            df = None
            cat_out = torch.cat([train, out_test],dim=0).cpu()
            scatters = umap_model.fit_transform(cat_out)
            df = pd.DataFrame(scatters, columns=["x", "y"])
            df ["label"] = None
            df ["label"][:length] = aug_mode[i]
            df ["label"][length:length*2] = test_mode[0]
            df ["label"][length*2:length*3] = test_mode[1]
            df ["label"][length*3:length*4] = test_mode[2]
            
            df_d = pd.concat([df[df["label"] == aug_mode[i]],df[df["label"] == test_mode[j]]], ignore_index=True)
            plt.figure()
            legend = True
            sns.scatterplot(data=df_d, x="x", y="y", hue="label", style='label',sizes='label', s=30 , palette=palette, legend = legend)
            plt.savefig(f"./figures_initcnn_aug_cho_0.5-0.333/{aug_mode[i]}--{test_mode[j]}.png", dpi=300)
            plt.close()

    print('done')


def buffer_sample_obs(buffer: object, batch_size: int = 1000):
    length= len(buffer._obses)
    idx = random.sample(range(length), batch_size)
    obses_samples =[buffer._obses[i] for i in idx]
    return obses_samples
def random_idx(length):
    idx = random.sample(range(length), length//2)
    return idx

if __name__ == '__main__':   
    main()