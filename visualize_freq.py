import pandas as pd
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib.pyplot import colorbar
from Tools.Visualize.VisdomPlotter import VisdomPlotter

H, W = 224, 224

def standard(data, dim=(-2, -1)):
    max_data, min_data = np.amax(data, axis=dim, keepdims=True), np.amin(data, axis=dim, keepdims=True)
    data = (data - min_data) / (max_data - min_data)
    return data

def acc_cnt_img(event, ord='txyp'):
    
    center = (260 // 2, 346 // 2)
    size = (224, 224)
    event = event[event[:, ord.find('x')] > (center[0] -  size[-2]// 2)]
    event = event[event[:, ord.find('x')] < (center[0] + size[-2] // 2)]
    event[:, ord.find('x')] -= (center[0] - size[-2] // 2)
    event = event[event[:, ord.find('y')] > (center[1] - size[-1] // 2)]
    event = event[event[:, ord.find('y')] < (center[1] + size[-1] // 2)]
    event[:, ord.find('y')] -= (center[1] - size[-1] // 2)

    x, y, p = np.split(event[:, (ord.find("x"), ord.find("y"), ord.find("p"))], 3, axis=1)
    
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    p = p.astype(bool)

    img = np.zeros((2, H * W))
    np.add.at(img[0], x[p] + W * y[p], 1.)
    np.add.at(img[1], x[~p] + W * y[~p], 1.)

    img = img.reshape((2, H, W))

    # normalize along the space dimension
    img = np.divide(img, 
                    np.amax(img, axis=(-2, -1), keepdims=True),
                    out=np.zeros_like(img),
                    where=img!=0)
    return img

def acc_cnt_clip(event, ord='txyp', T=5):
    center = (260 // 2, 346 // 2)
    size = (224, 224)
    event = event[event[:, ord.find('x')] > (center[0] -  size[-2]// 2)]
    event = event[event[:, ord.find('x')] < (center[0] + size[-2] // 2)]
    event[:, ord.find('x')] -= (center[0] - size[-2] // 2)
    event = event[event[:, ord.find('y')] > (center[1] - size[-1] // 2)]
    event = event[event[:, ord.find('y')] < (center[1] + size[-1] // 2)]
    event[:, ord.find('y')] -= (center[1] - size[-1] // 2)


    # event[:, ord.find('x')] *= (224 / 260)
    # event[:, ord.find('y')] *= (224 / 346)

    x, y, p, t = event[:, ord.find("x")], \
                event[:, ord.find("y")], \
                event[:, ord.find("p")], \
                event[:, ord.find("t")]
    
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    p = p.astype(bool)

    idx_before_bins = x \
                        + W * y \
                        + 0 \
                        + W * H * T * p
    num_voxels = int(W * H * 2 * T)
    clip = np.zeros(num_voxels)
    t *= T
    for i_bin in range(T):
        values =  np.maximum(1.0 - np.abs(t - i_bin), 0)
        # draw in voxel grid
        idx = idx_before_bins + W * H * i_bin
        np.add.at(clip, idx, values)
    clip = clip.reshape((2, T, H, W))
    
    return standard(clip)

class Time_Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
        self.pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    
    def forward(self, x):
        bx  = (x > 0).to(torch.float)
        bx = self.pad(self.pool(bx))
        corr = bx[:, :, :-1] * bx[:, :, 1:]
        corr = (corr > (5 / 9)).to(torch.float)
        return self.pool(corr)

class Time_Space_Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
        self.pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        k = 3
        kernel = torch.ones((2, 1, 1, k, k))
        kernel[..., k // 2, k // 2] = 0
        self.k = k
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # space filter
        bx = (x > 0).to(torch.float)
        around = F.conv3d(bx, self.weight, padding=(0, self.k // 2, self.k // 2), groups=2)
        # mean = self.avgpool(enhance)
        mean = self.avgpool(around)
        mask = (around < 5 * mean) & (x > 0)
        return ~mask & (x > 0)


if __name__ == '__main__':
    scene = 'l64'
    label = 5
    samples = pd.read_csv('Dataset/DAVISGait/train.csv', delimiter='\t')
    samples = samples[samples['light'] == scene]
    samples = samples[samples['label'] == label]
    samples = samples.reset_index(drop=True)

    # plotter = VisdomPlotter(env='visualize_freq', port=7000)
    dataset = h5py.File('Dataset/DAVISGait/C36W03.h5', 'r')

    item = samples.loc[11]
    data = dataset[item['light']][item['obj']][item['num']]
    data = np.stack([data[p] for p in 'txyp'], axis=-1).astype(float)
    data[:, 0] -= data[0, 0]
    data[:, 0] /= data[-1, 0]
    
    clip = acc_cnt_clip(data, T=5)

    plt.imshow(clip[0, 0])
    plt.title(scene)
    plt.show()

    # model = Time_Mask()
    # mask = model(torch.tensor(clip[None])).detach().numpy()[0]

    # FS = np.fft.fftn(enh_mat[0, 0])
    # enh_FS = np.log(np.abs(np.fft.fftshift(FS)) ** 2)

    # # plotter.heatMap(img_mat, win=scene)
    # plotter.images(raw_mat[None], win=f"raw {scene}")
    # plotter.images(enh_mat, win=f"enh {scene}")
    # plotter.heatMap(raw_FS, win=f"raw spectrum {scene}")

    # plotter.images(mask.transpose(1, 0, 2, 3), win=f"mask_{scene}")
    # mclip = 1 / (1 + np.exp(-clip)) - 0.5
    # mclip = standard(mclip * mask)
    # plotter.images(mclip.transpose(1, 0, 2, 3), win=f"mask_clip_{scene}") 
    # plotter.images(clip.transpose(1, 0, 2, 3), win=f"clip_{scene}")

    # model = importlib.import_module("Tools.Model.S2N_2D").Model({
    #                 'enhance':True,
    #                 'num_classes':36,
    #                 'in_channels':2,
    #                 'enhance_iter': 4,
    #                 'nIter': 3
    #             })
    # model.load_state_dict(torch.load('/home/wan97/Workspace/DVS/Recognition/S2N/Output/DAVISGait_S2N_2D_08021521/checkpoint.pkl'))
    # model.eval()

    # img_list = []
    # img_mat = np.zeros((H * 3, W * 3, 3))
    # for i in range(9):
    #     st = 1 / 9 * (i + 1)
    #     h, w = (i // 3) * H, (i % 3) * W
    #     img = acc_cnt_img(data[(data[:, 0] < st) & (data[:, 0] > st - 1 / 9)], ord='txyp')

    #     # FS = np.fft.fftn(img)
    #     # img_mat[h:h + H, w:w + W] = np.log(np.abs(np.fft.fftshift(FS)) ** 2)

    #     img_mat[h:h + H, w:w + W, 0] = img[0]
    #     img_mat[h:h + H, w:w + W, 1] = img[1]

    # img_list = []
    # for i in range(8):
    #     st = 1 / 8 * (i + 1)
    #     img = acc_cnt_img(data[(data[:, 0] < st) & (data[:, 0] > st - 1 / 8)], ord='txyp')

    #     # FS = np.fft.fftn(img)
    #     # img_mat[h:h + H, w:w + W] = np.log(np.abs(np.fft.fftshift(FS)) ** 2)
    #     img_list.append(img)
    
    # img_mat = torch.tensor(np.stack(img_list), dtype=torch.float)
    # raw_mat = acc_cnt_img(data)
    # with torch.no_grad():
    #     enh_mat = model.nsn(torch.tensor(raw_mat[None], dtype=torch.float))[0]
    # enh_mat = enh_mat.detach().numpy()

    # # frequency
    # FS = np.fft.fftn(raw_mat[0])
    # raw_FS = np.log(np.abs(np.fft.fftshift(FS)) ** 2)

    # FS = np.fft.fftn(enh_mat[0, 0])
    # enh_FS = np.log(np.abs(np.fft.fftshift(FS)) ** 2)

    # # plotter.heatMap(img_mat, win=scene)
    # plotter.images(raw_mat[None], win=f"raw {scene}")
    # plotter.images(enh_mat, win=f"enh {scene}")
    # plotter.heatMap(raw_FS, win=f"raw spectrum {scene}")
    # plotter.heatMap(enh_FS, win=f"enh spectrum {scene}")

    # fig = px.imshow(img_mat.astype(np.uint8), labels=dict(x=f"img_{scene}"), color_continuous_scale='viridis')
    # fig.update_layout(coloraxis_showscale=False)
    # fig.update_xaxes(showticklabels=False)
    # fig.update_xaxes(side="top")
    # fig.update_yaxes(showticklabels=False)
    # plotter.plotlyplot(fig, win=f"img_{scene}")

  

