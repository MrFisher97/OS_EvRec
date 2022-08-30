import torch.nn as nn
import os
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm
from .Enhance import Filter
# import ed3d as backbone
# import ed2d as backbone
import gait2d as backbone

MIN_VALUE = 1e-6

def standard(data, dim=(-2, -1)):
    max_data, min_data = torch.amax(data, dim=dim, keepdim=True), torch.amin(data, dim=dim, keepdim=True)
    data = (data - min_data) / (max_data - min_data)
    return data


class L_noise(nn.Module):
    '''
        Noise loss
        We coarsely identify the noise if it lacks adjacent event in k*k neighbor
    '''
    def __init__(self, k):
        super().__init__()
        self.k = k
        kernel = torch.ones((1, 2, 3, k, k)).cuda()
        kernel[..., 1, k//2, k//2] = 0
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, enhance):
        bool_enhance = (enhance > MIN_VALUE).to(torch.float)
        around = F.conv3d(bool_enhance, self.weight, padding='same').repeat(1, 2, 1, 1, 1)
        around[around < MIN_VALUE] = 0
        mask = (around < (3)) & (enhance > 0)

        # mean_val = torch.mean(enhance[enhance > 0])
        # around = F.conv2d(enhance, self.weight, padding=self.k // 2, groups=1).repeat(1, 2, 1, 1)
        # # mask = torch.logical_and(around == 0, enhance > 0)
        # mask = (around < (1e-5 * mean_val)) & (enhance > 0)
        # noise = torch.pow(enhance[mask], 2)
        area = torch.sum(mask, dim=(-2, -1))
        noise = torch.sum(enhance * mask, dim=(-2, -1))
        return torch.mean(noise / area), mask

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.noise = L_noise(5)
        self.classify = torch.nn.CrossEntropyLoss()

    def forward(self, feat, score, label, a):
        # l_noise, mask = self.noise(feat)
        l_cls = self.classify(score, label)

        loss = 0
        # loss += a * l_noise
        loss += (1 - a) * l_cls
        return {
                # 'mask':mask,
                'loss':{
                    'total':loss,
                    # 'noise':l_noise,
                    'class':l_cls,}
                }

class time_corr(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
        self.pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

    def forward(self, x):
        #  x = torch.sum(x, dim=1, keepdim=True)
        bool_x = (x > 0).to(torch.float)
        bool_x = self.pad(bool_x)
        # around = bool_x
        prev = self.pool(bool_x)
        around = prev[:, :, :-1] * bool_x[:, :, 1:]
        # corr[corr > MIN_VALUE] = 1
        around = (around > (4 / 9)).to(torch.float)
        return self.pool(around)
        # return bool_x

# class time_corr(nn.Module):
#     def __init__(self):
#         super().__init__()
#         k = 3
#         kernel = torch.ones((2, 1, 1, k, k))
#         kernel[..., k // 2, k // 2] = 0
#         self.k = k
#         self.weight = kernel.cuda()
#         self.avgpool = nn.AdaptiveAvgPool3d(1)


#     def forward(self, x):
#         #  x = torch.sum(x, dim=1, keepdim=True)
#         bx = (x > 0).to(torch.float)
#         around = F.conv3d(bx, self.weight, padding=(0, self.k // 2, self.k // 2), groups=2)
#         # mean = self.avgpool(enhance)
#         mean = self.avgpool(around)
#         mask = (around < 5 * mean) & (x > 0)
#         return ~mask & (x > 0)

class space_corr(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)
        x = (x > 0).to(torch.float)
        x = self.pool(x) - x / 9
        x[x< MIN_VALUE] = 0
        # mask = (around < (3 / 9)) & (x > 0)
        x = x > (2 / 9)
        # corr[corr > MIN_VALUE] = 1
        return x.to(torch.float)


class Dynamic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, k=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = tuple(map(lambda x:x // 2, self.kernel_size))
        self.k = k

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//2, bias=False)
        self.fc2 = nn.Linear(in_channels//2, k, bias=False)
        self.activation = nn.ReLU()

        # self.norm = nn.BatchNorm3d(out_channels)
        self.dynamic_weight = nn.Parameter(torch.randn((k, out_channels, in_channels, *kernel_size)))
        nn.init.kaiming_uniform_(self.dynamic_weight)


    def forward(self, x):
        a = self.global_pool(x)[..., 0, 0, 0]
        a = self.activation(self.fc1(a))
        a = F.softmax(self.fc2(a), dim=1)

        B, C, T ,H, W = x.size()
        x = x.reshape(1, -1, T, H, W)

        weight = torch.einsum('ij, jk... -> ik...', a, self.dynamic_weight).view(-1, self.in_channels, *self.kernel_size)
        x = F.conv3d(x, weight=weight, padding=self.padding, groups=B)
        x = x.reshape(B, -1, T, H, W)
        return x


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        super().__init__()
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
            self.to(torch.float)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = F.relu(x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)
            
            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()
            optim.zero_grad()

        path = join(dirname(__file__), "quantization_layer_init")
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), join(path, "trilinear_init.pth"))


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

class QuantizationLayer(nn.Module):
    def __init__(self, size,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.ReLU(),
                 event_embed = 'tyxpb'):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=size[0])
        self.size = size
        self.event_embed = event_embed

    def base_encoding(self, events):
        x, y, t, p, b = events[:, self.event_embed.index('x')], \
                        events[:, self.event_embed.index('y')], \
                        events[:, self.event_embed.index('t')], \
                        events[:, self.event_embed.index('p')], \
                        events[:, self.event_embed.index('b')]
    
        T, H, W = self.size
        B = int((1 + b[-1]).item())
        num_pixels = int(W * H * 2 * B)
        # idx = x + W * y + W * H * p + W * H * 2 * b
        # event_cnt = events[0].new_full([num_pixels,], fill_value=0)
        # event_cnt.put_(idx.long(), torch.ones_like(x), accumulate=True)
        # event_cnt = event_cnt.view(-1, 2, H, W)[:, (1, 0)]

        # event_mask = torch.sum(event_cnt, dim=1, keepdims=True)
        # event_mask = event_mask > 0

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * T * p \
                          + W * H * T * 2 * b
        num_voxels = num_pixels * T
        event_vox = events[0].new_full([num_voxels,], fill_value=0.)
        t *= T 
        for i_bin in range(T):
            values =  torch.clamp(1.0 - torch.abs(t - i_bin), min=0)
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            event_vox.put_(idx.long(), values, accumulate=True)

        t /= T
        event_vox = event_vox.view(-1, 2, T, H, W)

        return {
            # 'event_cnt': event_cnt,
            # 'event_mask': event_mask,
            'event_vox': event_vox.float(),
        }

    def learn_encoding(self, events):
        x, y, t, p, b = events[:, self.event_embed.index('x')], \
                        events[:, self.event_embed.index('y')], \
                        events[:, self.event_embed.index('t')], \
                        events[:, self.event_embed.index('p')], \
                        events[:, self.event_embed.index('b')]
    
        T, H, W = self.size
        B = int((1 + b[-1]).item())

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * T * p \
                          + W * H * T * 2 * b
        num_voxels = int(2 * np.prod(self.size) * B)
        time_vox = events[0].new_full([num_voxels,], fill_value=0.)
        
        for i_bin in range(T):
            values = self.value_layer.forward(t - i_bin/(T - 1))
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin

            time_vox.put_(idx.long(), t * values, accumulate=True)

        time_vox = time_vox.view(-1, 2, T, H, W)
        # a = time_vox.cpu().detach().numpy()
        # time_vox = self.conv(time_vox)
        return {
            'time_vox': time_vox,
            # 'embed_vox': standard(embed_vox, dim=(-2, -1)), 
        }

    def forward(self, events):
        # points is a list, since events can have any size
        bi = self.event_embed.index('b')
        B = int((1+events[-1, bi]).item())

        # get values for each channel
        # x, y, t, p, b = events.t()
        
        # normalizing timestamps
        ti = self.event_embed.index('t')
        for b in range(B):
            events[:, ti][events[:, bi] == b] /= events[:, ti][events[:, bi] == b].max()
        
        # pi = self.event_embed.index('p')
        # events[:, pi] = (events[:, pi] + 1) / 2  # maps polarity to 0, 1

        base_encoding = self.base_encoding(events)
        # learn_encoding = self.learn_encoding(events)

        return base_encoding.get('event_vox', None)
        # return {'event_vox': base_encoding.get('event_vox', None),
                # 'time_vox': learn_encoding.get('time_vox', None),
                # 'embed_vox': learn_encoding.get('embed_vox', None),
                # }

class Time_filter(nn.Module):
    def __init__(self, in_channels, voxel_size):
        super(Time_filter, self).__init__()
        self.op = nn.ModuleDict({
                'corr':time_corr(),
                'conv1':nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(1, 1, 1), bias=False),
                # 'conv2':nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
                'conv2':Dynamic_Conv(in_channels=16, out_channels=16, kernel_size=(3, 1, 1)),
                'conv3':nn.Conv3d(in_channels=16, out_channels=in_channels, kernel_size=(1, 1, 1), bias=False),
                'act':nn.Sigmoid(),
                # 'norm':nn.LayerNorm(voxel_dim[-2:]),
                'time_agg':Dynamic_Conv(in_channels=voxel_size[0], out_channels=voxel_size[0], kernel_size=(1, 1, 1)),
                'norm':nn.InstanceNorm3d(2),
        })

    def forward(self, x):
        mask = self.op['corr'](x)
        x = self.op['norm'](x)
        x = self.op['conv1'](x)
        x = self.op['conv2'](x)
        x = self.op['conv3'](x)
        # x = self.op['time_agg'](x.transpose(2, 1)).transpose(2, 1)
        x = self.op['act'](x) * mask
        x = self.op['norm'](x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=10, in_channels=2, nIter=3, size=[9, 128, 128], 
                    mlp_layers=[1, 30, 30, 1], activation=nn.LeakyReLU(0.1), **kwargs):
        super().__init__()
        self.quantization = QuantizationLayer(size, mlp_layers, activation, event_embed='txypb')

        self.encoder = nn.ModuleDict({
            'tfilter': Time_filter(in_channels=in_channels, voxel_size=size),
            'sfilter': nn.Sequential(
                                    Filter.Anistropic_Diffusion_3D(in_channels=in_channels, kernel_size=1, sigma=1, nIter=nIter),
                                    nn.LayerNorm(size[-2:]),
                                    ),
            'backbone': backbone.Model(num_classes=num_classes, in_channels=in_channels * size[0]),
        })

    def forward(self, events=None):
        embed_vox = self.quantization(events)
        embed_vox = self.encoder['tfilter'](embed_vox)
        embed_vox = self.encoder['sfilter'](embed_vox)
        embed_vox = torch.cat([embed_vox[:, 0], embed_vox[:, 1]], 1)
        # embed_vox = standard(self.filter(embed_vox), dim=(-2, -1))
        score = self.encoder['backbone'](embed_vox)
        return score
        # return {
        #     'event_vox': embed.get('event_vox', None),
        #     'time_vox': embed.get('time_vox', None),
        #     'embed_vox': torch.stack([embed_vox[:, :9], embed_vox[:, 9:]], 1),
        #     'score': score,
        # }