#!/bin/bash
# python train.py --config DVSGesture_S2N --override 'Data.scene=lab, Data.data_file=C11W05.h5'

source /home/wan97/Software/miniconda3/bin/activate ros
python train.py --config DAVISGait_tmp2d_TSN --override 'Train.num_epochs=20, Model.theta=0.4'
# python train.py --config DAVISGait_tmp2d_TSN --override 'Train.num_epochs=20, Model.theta=0.666, Model.mask_kernel=3'
# python train.py --config DAVISGait_tmp2d_TSN --override 'Train.num_epochs=20, Model.theta=0.777, Model.mask_kernel=3'
# python train.py --config DAVISGait_tmp2d_TSN --override 'Train.num_epochs=20, Model.theta=0.555, Model.mask_kernel=5'
# python train.py --config DAVISGait_tmp2d_TSN --override 'Train.num_epochs=20, Model.theta=0.555, Model.mask_kernel=7'