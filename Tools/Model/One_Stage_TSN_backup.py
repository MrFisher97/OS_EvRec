import torch.nn as nn
import torch
import torch.nn.functional as F
from NonLocal import NonLocalBlockND

MIN_VALUE = 1e-6
MAX_VALUE = 1e6

def standard(data, dim=(-2, -1)):
    max_data, min_data = torch.amax(data, dim=dim, keepdim=True), torch.amin(data, dim=dim, keepdim=True)
    data = (data - min_data) / (max_data - min_data)
    data[torch.isnan(data)] = 0
    return data

class Time_Space_Corr(nn.Module):
    def __init__(self, theta=3/9, mask_kernel=3):
        super().__init__()
        self.k = int(mask_kernel)
        kernel = torch.ones((2, 1, 1, self.k, self.k))
        kernel[..., self.k // 2, self.k // 2] = 0
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
        self.pool = nn.AvgPool3d(kernel_size=(1, self.k, self.k), stride=1, padding=(0, self.k // 2, self.k // 2))

        self.theta = nn.Parameter(data=torch.tensor(theta), requires_grad=False)


    def forward(self, x):
        # space filter
        bx = (x > 0).to(torch.float)
        around = F.conv3d(bx, self.weight, padding=(0, self.k // 2, self.k // 2), groups=2)
        # mean = self.avgpool(enhance)
        mean = self.avgpool(around)
        smask = (around < 5 * mean) & (x > 0)

        # time filter
        bx = self.pad(bx)
        # around = bool_x
        prev = self.pool(bx)
        around = prev[:, :, :-1] * bx[:, :, 1:]
        # corr[corr > MIN_VALUE] = 1
        tmask = (around > self.theta)

        mask = tmask | (~smask & (x > 0))
        # mask = tmask
        # mask = ~smask & (x > 0)
        return self.pool(mask.to(torch.float))

class Anistropic_Diffusion_3D(nn.Module):
    def __init__(self, in_channels, nIter=3, sigma=1, **kwargs):
        super().__init__()
        direction_weights = []

        self.max_sigma = sigma
        self.num_direction = 8
        self.in_channels = in_channels
        self.nIter = nIter

        if self.num_direction == 4:
            direction = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        elif self.num_direction == 8:
            direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(self.num_direction):
            kernel = torch.zeros((1, 3, 3)) 
            kernel[0, 1, 1] = -1
            kernel[0, 1 + direction[i][0], 1 + direction[i][1]] = 1
            direction_weights.append(kernel)

        direction_weights = torch.stack(direction_weights, dim=0)
        direction_weights = direction_weights.unsqueeze(1).tile(in_channels, 1, 1, 1, 1)

        self.weight = nn.Parameter(direction_weights, requires_grad=False)
        # self.weight = direction_weights
        conv_weight = torch.ones((2, self.num_direction, 1, 1, 1)) / 4
        self.conv_weight = nn.Parameter(conv_weight, requires_grad=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        # get the gradient
        xGrad = F.conv3d(x, weight=self.weight, padding=(0, 1, 1), groups=self.in_channels)

        # Get the sigma
        sigma = self.global_pool(torch.abs(xGrad))
        B, C, T = sigma.size()[:3]
        sigma = sigma.view(B, C//self.in_channels, self.in_channels, T, 1, 1).softmax(dim=1)

        # clamp the sigma
        sigma = torch.clamp(sigma.view(B, C, T, 1, 1) * 8, min=0.01)

        for i in range(self.nIter):
            gWeight = torch.exp(- torch.pow(xGrad, 2) / sigma)
            diffusion = xGrad * gWeight
            diffusion = F.conv3d(diffusion, weight=self.conv_weight, groups=self.in_channels)
            diffusion = self.activation(diffusion)
            x = x + diffusion
            if self.nIter > 1:
                xGrad = F.conv3d(x, weight=self.weight, padding=1, groups=self.in_channels)
        # return normalize(x, dim=(2, 3))
        return x

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
        self.dynamic_weight = nn.Parameter(torch.empty((k, out_channels, in_channels, *kernel_size)))
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

class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1, input_channels=512, num_segments=5, num_classes=36):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.num_segments = num_segments
        self.sensor = nn.ModuleDict({
            'head': nn.Sequential(
                        nn.Linear(input_channels, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(512, num_classes),
                        )
        })
        if self.consensus_type == 'rnn':
            self.sensor.update({
                            'rnn': nn.LSTM(input_size=input_channels, hidden_size=input_channels, 
                                  bidirectional=False, num_layers=1, bias=False, batch_first=False),
                            })
        elif self.consensus_type == 'atten':
            self.sensor.update({
                            'atten': NonLocalBlockND(in_channels=input_channels, inter_channels=input_channels, sub_sample=False, dimension=1)
            })

    def forward(self, inp):
        if self.consensus_type == 'rnn':
            inp = inp.view(-1, self.num_segments, inp.size(-1))
            out = self.sensor['rnn'](inp)
            out = self.sensor['head'](out[0])
            out = out.mean(dim=1)
        elif self.consensus_type == 'avg':
            out = self.sensor['head'](inp)
            out = out.view((-1, self.num_segments, out.size(-1)))
            out = out.mean(dim=1)
        elif self.consensus_type == 'atten':
            inp = inp.view(-1, self.num_segments, inp.size(-1))
            inp = inp.permute(0, 2, 1)
            out = self.sensor['atten'](inp)
            out = out.permute(0, 2, 1)
            out = self.sensor['head'](out)
            out = out.mean(dim=1) 
        return out                

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)

def convert_temporal_shift(net, n_segment, n_div, place='BasicBlock'):
    for name, mod in list(net.named_children()):
        convert_temporal_shift(mod, n_segment, n_div, place)
        if place in str(type(mod)):
            # blocks = list(mod.children())
            # for i, b in enumerate(blocks):
            #     blocks[i] = TemporalShift(b, n_segment=n_segment, n_div=n_div)
            # return nn.Sequential(*(blocks))
            # setattr(net, name, nn.Sequential(*(blocks)))
            setattr(net, name, TemporalShift(mod, n_segment=n_segment, n_div=n_div))

class DCD_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.1, **kwargs):

        super(DCD_Conv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                        stride, padding, dilation, groups, bias)
        self.theta = theta
        self.pool_pad = list(map(lambda x: x // 2, self.kernel_size))

    def forward(self, x):
        out = super().forward(x)
        kernel_diff = self.weight.sum((2, 3), keepdim=True)
        
        mask = F.avg_pool2d((x > 0).to(torch.float), kernel_size=self.kernel_size,
                        padding=self.pool_pad, stride=1)                       

        out_diff = F.conv2d(input=x * mask, weight=kernel_diff, bias=self.bias, 
                        stride=self.stride, padding=0, groups=self.groups)

        return out - self.theta * out_diff

def convert_CDC(net, theta=0.3):
    for name, mod in list(net.named_children()):
        convert_CDC(mod, theta)
        if type(mod) == nn.Conv2d:
            mod.theta = theta
            setattr(net, name, DCD_Conv2d(**mod.__dict__))

class Time_filter(nn.Module):
    def __init__(self, in_channels, voxel_size, theta=3/9, mask_kernel=3):
        super(Time_filter, self).__init__()
        self.op = nn.ModuleDict({
                'corr':Time_Space_Corr(theta=theta, mask_kernel=mask_kernel),
                # 'conv1':nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(1, 1, 1), bias=False),
                'conv2':Dynamic_Conv(in_channels=2, out_channels=2, kernel_size=(3, 1, 1)),
                # 'act':nn.Sigmoid(),
                'act':nn.Tanh(),
                # 'norm':nn.LayerNorm(voxel_dim[-2:]),
                'norm':nn.InstanceNorm3d(2),
        })

    def forward(self, x):
        mask = self.op['corr'](x)
        x = self.op['norm'](x)
        # x = standard(x)
        x = self.op['conv2'](x)
        # x = x * mask
        x = self.op['act'](x)
        x = standard(x)
        x = x * mask
        return x


class Model(nn.Module):
    def __init__(self, num_classes, in_channels, nIter=1, size=[5, 224, 224], 
                       consensus_type='avg', is_shift=False, shift_div=8, backbone='gait2d',
                       is_CDC=True, CDC_theta=0.1, mask_theta=3/9, mask_kernel=3, **kwargs):
        super().__init__()

        if backbone == 'gait2d':
            import gait2d
            backbone_model = gait2d.Model(num_classes=num_classes, in_channels=in_channels)
            backbone_model.name = 'gait2d'
            last_layer_name = 'classifier'
            feature_dim = getattr(backbone_model, last_layer_name)[0].in_features
            temporal_shift_block = 'ResidualBlock'
        elif backbone == 'resnet34':
            import torchvision.models.resnet as resnet
            backbone_model = resnet.resnet34(pretrained=False)
            last_layer_name = 'fc'
            backbone_model.name = 'resnet34'
            feature_dim = getattr(backbone_model, last_layer_name).in_features
            temporal_shift_block = 'BasicBlock'
            conv1 = getattr(backbone_model, 'conv1')
            setattr(backbone_model, 'conv1', nn.Conv2d(2, conv1.out_channels, conv1.kernel_size, conv1.stride, conv1.padding, conv1.dilation, conv1.groups, conv1.bias))

        self.encoder = nn.ModuleDict({
            'tfilter': Time_filter(in_channels=in_channels, voxel_size=size, theta=mask_theta, mask_kernel=mask_kernel),
            'sfilter': nn.Sequential(
                                    Anistropic_Diffusion_3D(in_channels=in_channels, nIter=nIter),
                                    nn.LayerNorm(size[-2:]),
                                    ),
            'backbone': backbone_model,
            'consen': ConsensusModule(consensus_type, dim=1, input_channels=1024, num_segments=size[0], num_classes=num_classes)
        })

        if is_CDC:
            print('Converting the Conv to CDConv')
            convert_CDC(self.encoder['backbone'], theta=CDC_theta)

        if is_shift:
            print(f'Converting the {temporal_shift_block} to temporal-shift module...')
            convert_temporal_shift(self.encoder['backbone'], size[0], n_div=shift_div, place=temporal_shift_block)

        last_layer = nn.Sequential(
                                nn.Linear(feature_dim, 1024),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(1024),
                                nn.Dropout(p=.5),
                                )
    
        setattr(self.encoder['backbone'], last_layer_name, last_layer)

    def forward(self, x):
        # x = self.encoder['tfilter'](x)
        # x = self.encoder['sfilter'](x)

        x = x.transpose(2, 1)
        x = x.reshape(-1, *x.size()[2:])

        x = self.encoder['backbone'](x)
        x = self.encoder['consen'](x)
        return x

    
if __name__ == '__main__':
    import numpy as np
    import random

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)

    config = {
            "num_classes": 36,
            "in_channels": 2, 
            "nIter": 1,
            "consensus_type": 'rnn',
            "is_shift": True, 
            "shift_div": 4,
            "size": [5, 224, 224], 
            "backbone": 'gait2d',
            "is_CDC": True,
            "CDC_theta": 0.1,
            "mask_theta": 3/9, 
            "mask_kernel": 3,
            }

    net = Model(**config)

    # for c in net.children():
    #     print(c)

    params = np.sum([p.numel() for p in net.encoder['backbone'].parameters()]).item()
    params = params * 4 // 1024 // 1024
    print(f"Loaded parameters : {params:.3e} M")

    params = np.sum([p.numel() for p in net.parameters()]).item()
    params = params * 4 // 1024 // 1024
    print(f"Loaded parameters : {params:.3e} M")

    a = torch.ones(5, 2, 5, 224, 224)
    print(net(a))