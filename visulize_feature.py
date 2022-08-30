# # -- coding: utf-8 --**
import os
# import sys
# sys.path.append("..")

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import importlib
from torchvision import transforms
import argparse
import json
import logging
import time
from Tools.Visualize.VisdomPlotter import VisdomPlotter
from torch.utils.tensorboard import SummaryWriter
import logging

import Tools.utils as utils

MAX_VALUE = 1e8

def standard(data, axis=(-2, -1)):
    max_data, min_data = np.amax(data, axis=axis, keepdims=True), np.amin(data, axis=axis, keepdims=True)
    delta = max_data - min_data
    delta[delta == 0] = MAX_VALUE
    data = (data - min_data) / delta
    return data

class Visual_Session(utils.Session):
    def build_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()

        if not logger.handlers:
            logger.addHandler(ch)

        self.logger = logger
        recorder = self.config['Recorder']
        self.plotter = VisdomPlotter(env=f"{self.config['Model']['name']}", port=7000)
        self.writer = SummaryWriter('Runs') if recorder['show_tensorboard'] else None

    def _build_model(self):
        super()._build_model()
        self.net = importlib.import_module(f"Tools.Model.{self.config['Model']['name']}").Model(**self.config['Model'])
        self.net = self.net.to(self.device)
        self.net.load_state_dict(torch.load(os.path.join(self.config['log_dir'], 'checkpoint.pkl')))
        self.criterion = torch.nn.CrossEntropyLoss()

    def _batch(self, item, show=False):
        data, label = item['data'].to(self.device, dtype=torch.float), item['label'].to(self.device)
        with autocast():
            output = self.net(data)
            if isinstance(output, dict):
                score = output['score']
            else:
                score = output
            loss = self.criterion(score, label)

            if self.plotter and show:
                raw_vox = output['event_vox']
                B, C, T, H, W = raw_vox.size()
                nframe = np.random.randint(0, T, size=B)
                raw_vox = raw_vox[np.arange(B), :, nframe]
                self.plotter.images(raw_vox.detach().cpu().numpy(), f"Enhance_raw@{self.cur_scene}", if_standard=True)
                
                embed_vox = output.get('embed_vox', None)
                if embed_vox is not None:
                    embed_vox = embed_vox[np.arange(B), :, nframe]
                    self.plotter.images(embed_vox.detach().cpu().numpy(), f"Enhance_result@{self.cur_scene}", if_standard=True)

                mask = loss.get('mask', None)
                if mask is not None:
                    mask = mask[np.arange(B), :, nframe]
                    self.plotter.images(mask.detach().cpu().numpy(), f"Mask_result@{self.cur_scene}")
            
        pred = score.max(1)[1]
        return {'loss':loss.cpu(), 'pred':pred.cpu()}

    def _epoch(self, data_loader, show=False):
        loss = utils.Param_Detector()
        acc = utils.Param_Detector()
        time = utils.Time_Detector()
        class_pred = utils.Category_Detector(self.config['Data']['num_classes'])
        
        for i, item in enumerate(tqdm(data_loader, ncols=80)):
            result = self._batch(item)
            loss.update(result['loss'])
            acc.update(result['pred'].eq(item['label']).sum(), item['label'].size(0))
            time.update(item['label'].size(0))
            class_pred.update(result['pred'], item['label'])
       
        print(f"{self.config['Data']['scene']} : loss:{loss.avg:.3f}, acc:{acc.avg:.1%}, {time.avg:.6f}  seconds/batch")
        return {'loss': loss.avg,
                'acc':acc.avg,
                'class_acc':class_pred.val,
                'time':time.avg}
    
    def _enhance_batch(self, item):
        with autocast():
            data = item['data'].to(self.device)

            if self.config['Model']['name'] == 'Co_Model_3D':
                B, T = data.size(0), data.size(2)
                nframe = np.random.randint(0, T, size=B)
                data = data.permute(0, 2, 1, 3, 4)
                data = data[np.arange(B), nframe]
            
            enhance_result, E = self.net.enhance(data)
            enhance_loss = self.enhance_criterion(enhance_result, E, data)
            loss = enhance_loss['total']

        if self.plotter:
            self.plotter.images(data.detach().cpu().numpy(), win=f"Enhance_train_raw@{self.config['Data']['scene']}")
            self.plotter.images(enhance_result.detach().cpu().numpy(), win=f"Enhance_train_result@{self.config['Data']['scene']}", )
        return enhance_loss

    def visualize(self):
        # get feature
        self.net.eval()
        feat_list = []
        def hook_fn_forward(module, inp, oup):
            print(module)
            # print(inp[0].size())
            # feat_list.append(inp[0].detach().cpu().numpy())
            print(oup.size())
            feat_list.append(oup.detach().cpu().numpy())

        
        # module register
        # self.net.model.Conv3d_1a_7x7.register_forward_hook(hook_fn_forward) # I3D        
        # self.net.backbone.Conv3d_2b_1x1.register_forward_hook(hook_fn_forward) # S2N
        # self.net.fsn.DCDC.features[0].register_forward_hook(hook_fn_forward) # S2N2D
        # self.net.encoder.backbone.features[0].register_forward_hook(hook_fn_forward) # tmp2D
        # self.net.base_model.conv1.register_forward_hook(hook_fn_forward) # tmp2D
        # self.net.encoder.tfilter.op['conv2'].register_forward_hook(hook_fn_forward) # tmp2D
        # self.net.encoder[0].op['conv2'].register_forward_hook(hook_fn_forward) # tmp2D
        self.net.encoder['tfilter'].register_forward_hook(hook_fn_forward) # Tmp

        for scene in ['l0', 'l4', 'l16', 'l64']:
            self.config['Data']['scene'] = scene
            self.config['Data']['data_file'] = "C36W03.h5"
            test_dataset = self._load_data('Train').dataset
            feat_list = []
            with torch.no_grad():
                nsample = 330
                item = test_dataset[nsample]
                data, label = item['data'], torch.as_tensor([item['label']])
                data = data[None, ...]
                # data = torch.concat([data, torch.zeros(data.size(0), 1)], 1)
                data, label = data.to(self.device), label.to(self.device)
                oup = self.net(data)
                score = oup['score'] if isinstance(oup, dict) else oup
                loss = self.criterion(score, label)
            
            # concate output 4 x 4 ( T x C )

            # plot feature
            # img = []
            # c = 1
            # var = 0
            # mean = 0
            # for i in range(c, c + 4):
            #     tmp = []
            #     for j in range(1, 5):
            #         var += np.var(feat_list[0][0, i * 4 + j])
            #         mean += np.mean(feat_list[0][0, i * 4 + j])
            #         tmp.append(standard(np.flip(feat_list[0][0, i * 4 + j])))
            #     img.append(np.concatenate(tmp, axis=0))
            # img = np.concatenate(img, axis=1)
            # self.plotter.heatMap(img, win=f"test_N{nsample}_C{c}_S{self.config['Data']['scene']}")


            # plot FFT
            C = 1
            img = np.flip(feat_list[0][0, 0, C])
            self.plotter.heatMap(img, win=f"Img N{nsample}_S{self.config['Data']['scene']}_M{np.mean(img[img > 0]):.3f}_V{np.var(img[img > 0]):.3f}")

            # min_val = np.min(img)
            # self.plotter.histogram(img.flatten(), win=f"DIst N{nsample}_S{self.config['Data']['scene']}")

            # FS = np.fft.fftn(img)
            # FS = np.log(np.abs(np.fft.fftshift(FS)) ** 2)
            # self.plotter.heatMap(FS, win=f"Spectrum N{nsample}_S{self.config['Data']['scene']}")

    def visualize_point_cloud(self):
        from sklearn.utils import resample

        self.config['Data']['scene'] = 'fluorescent'
        self.config['Data']['data_file'] = "C11W05.h5"

        test_dataset = self._load_data('Train', transforms=None).dataset
        for nsample in range(2030, 3000):
            # nsample = 2000
            # print(test_dataset.samples.loc[nsample])
            item = test_dataset[nsample]
            data, label = item['data'], torch.tensor([item['label']])
            if label == 8:
                break
        npoint = 8092
        data = resample(data, n_samples = npoint, random_state=2022)

        markercolor = np.ones((data.size(0), 3))
        markercolor[data[:, -1] == 0] *= (0, 200, 200)
        markercolor[data[:, -1] == 1] *= (200, 0, 100)
        self.plotter.scatter(data[:, :3], win=f"test_S{self.config['Data']['scene']}_N{nsample}_P{npoint}", color=markercolor)


    def test(self):
        self.plotter = None
        self.net.eval()
        if self.plotter:
            self.plotter.text(f"Test @ {self.config['Data']['dataset']}", win=f"Test @ {self.log_dir}")
        with torch.no_grad():
            for scene in self.config['Test']['scenes']:
                self.cur_scene = scene
                self.config['Data']['scene'] = scene
                test_loader = self._load_data('Test')
                test_result = self._epoch(test_loader, show=False)

                if self.plotter:
                    test_info = f"@ {scene}, loss:{test_result['loss']:.3f}, acc:{test_result['acc']:.1%}, {test_result['time']:.6f}  seconds/batch\n"
                    self.plotter.text(test_info, win=f"Test @ {self.log_dir}")

                if self.writer:
                    test_info = f"@ {scene}, loss:{test_result['loss']:.3f}, acc:{test_result['acc']:.1%}, {test_result['time']:.6f}  seconds/batch"
                    self.writer.add_text('Test', test_info)
                # compute the class accuracy and print
                # class_acc = test_result['class_acc'] / np.sum(test_result['class_acc'], axis = 1, keepdims=True)
                # s = ' ' * 30
                # for i in range(self.config['num_classes']):
                #     s += f'{i:^8d}'
                # self.logger.info(s)
                # for i in range(self.config['num_classes']):
                #     s = f'{self.label_list[i]:^30}' # control the print form
                #     for j in range(self.config['num_classes']):
                #         val = f'{class_acc[i, j]:.3f}'
                #         s += f'{val:^8}'
                #     self.logger.info(s)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='DAVISGait_tmp2d_TSN')
    args.add_argument('--log_dir', type=str, default='Output/DAVISGait_tmp2d_TSN_08291133')
    args = vars(args.parse_args())
    config = json.load(open(f"Tools/Config/{args['config']}.json", 'r'))
    config['log_dir'] = args['log_dir']
    # exit(0)
    sess = Visual_Session(config)
    sess._build_model()
    # sess.test()
    sess.visualize()
    # sess.visualize_point_cloud()
    # sess.close()

    # print model architecture
    # net = Model.Net(imsize=(256, 256), in_size=2, num_class=8)
    # print(net)