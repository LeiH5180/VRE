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
from augmentations import random_conv, random_overlay, random_shift
from umap import UMAP
from resnet_model_test import resnet_model
from arguments import parse_args


def main():
    args = parse_args()
    de_num = args.de_num
    replay_buffer = torch.load('saved_buffers/replay_buffer_500000.pt')
    test_buffer_1 = torch.load('saved_buffers/test_buffer_1_500000.pt')
    test_buffer_2 = torch.load('saved_buffers/test_buffer_2_500000.pt')
    test_buffer_3 = torch.load('saved_buffers/test_buffer_3_500000.pt')
    # RL更新所用的frame是9*84*84，一次采了三张图片
    batch_size = 1000
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
        rb_conv = random_conv(rb_tensor)
        rb_over = random_overlay(rb_tensor)

    # ---------resnet&umap----------------
    RS = resnet_model('resnet50')
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, densmap = True, dens_lambda = 2)

    with torch.no_grad():
        out_rb = RS.resnet_model(rb_tensor)
        out_shift = RS.resnet_model(rb_shift)
        out_conv = random_conv(rb_conv)
        out_over = random_overlay(rb_over)
        out_tb1 = RS.resnet_model(tb1_tensor)
        out_tb2 = RS.resnet_model(tb2_tensor)
        out_tb3 = RS.resnet_model(tb3_tensor)
    
    length = out_rb.shape[0]
    all_out = torch.cat([out_rb,out_shift,out_conv,out_over,out_tb1,out_tb2,out_tb3],dim=0).cpu()
    # all_out = np.vstack([out_rb.cpu().numpy(),out_tb1.cpu().numpy(),out_tb2.cpu().numpy(),out_tb3.cpu().numpy()])    # for np.array
    scatters = umap_model.fit_transform(all_out)

    # ------------plot--------------------
    df = pd.DataFrame(scatters, columns=["x", "y"])
    df ["label"] = None
    df ["label"][:length] = "train_ori"
    df ["label"][length:2*length] = "train_shift"
    df ["label"][2*length:3*length] = "train_conv"
    df ["label"][3*length:4*length] = "train_over"    
    df ["label"][4*length:5*length] = "color_hard"
    df ["label"][5*length:6*length] = "video_easy"
    df ["label"][6*length:] = "video_hard"

    palette = sns.color_palette("Set1",5)
    sns.set_palette(palette)
    aug_mode = ["train_state","train_shift","train_conv","train_over"]
    test_mode = ["color_hard","video_easy","video_hard"]
    palette=['#DC565F', '#3E61AC']
    for j in range(4):
        for i in range(3):       
            df_d = pd.concat([df[df["label"] == aug_mode[j]],df[df["label"] == test_mode[i]]], ignore_index=True)
            df_d.to_csv("encoding.csv")
            plt.figure()
            legend = True
            sns.scatterplot(data=df_d, x="x", y="y", hue="label", style='label',sizes='label', s=20 , palette=palette, legend = legend)
            
            plt.savefig(f"{aug_mode}--{test_mode[i]}.png", dpi=300)
            plt.close()


    # frame = np.transpose(test_buffer_3._obses[0][1].frames[0],(1,2,0))
    # plt.imshow(frame)
    # plt.imsave('test_buffer_3_frame_0.png', frame)




    # TODO: T-SNE or UMDP

def buffer_sample_obs(buffer: object, batch_size: int = 1000):
    length= len(buffer._obses)
    idx = random.sample(range(length), batch_size)
    obses_samples =[buffer._obses[i] for i in idx]
    return obses_samples

if __name__ == '__main__':
    main()