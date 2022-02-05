# python imports
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

from skimage.transform import resize
import math

sys.path.append('../../ext/medipy-lib')
sys.path.append('../../src')

from model import cvpr2018_net, SpatialTransformer
from medipy.metrics import dice
from utils import get_names,name2vol,relabel_CSF,relabel
from plugin import plugin_dst, create_grid, create_dst_matrix



def test(gpu, 
         atlas_file, 
         model_load,
         reduc_size,
         model,
         test_file,
         slurm_data_path):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from or the model itself
    :param reduc_size: whether to reduce size of volumes
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    

    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    
    atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
    if reduc_size == 1:
        atlas_vol = resize(atlas_vol, (1,160/2,192/2,224/2,1))
        
    atlas_seg = atlas['seg']
    if reduc_size == 1:
        atlas_seg = np.round(resize(atlas_seg, (160/2,192/2,224/2)))
        
    vol_size = atlas_vol.shape[1:-1]
    
    if isinstance(model_load, str):
        # Prepare the vm1 or vm2 model and send to device
        nf_enc = [16, 32, 32, 32]
        if model == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif model == "vm2":
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        # Set up model
        model = cvpr2018_net(vol_size, nf_enc, nf_dec)
        model.to(device)
        checkpoint = torch.load(model_load)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model_load
    

    # Test file and anatomical labels we want to evaluate
    test_vol_names, test_seg35_names, test_seg4_names = get_names(test_file,slurm_data_path,seg=True)


    good_labels=[2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,26,28,30,31,41,42,43,44,46,47,48,50,51,52,53,54,58,60,62,63]

         
    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol).to(device).float()
    input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)
    B,C,D,H,W = input_fixed.shape

    # Use this to warp segments
    trf = SpatialTransformer(atlas_vol.shape[1:-1], mode='nearest')
    trf.to(device)
    

    for k in range(0, len(test_vol_names)):
        
        X_vol = name2vol(test_vol_names[k]  , reduc_size, seg=False)
        X_seg35 = name2vol(test_seg35_names[k], reduc_size, seg=True)
        X_seg4 = name2vol(test_seg4_names[k] , reduc_size, seg=True)
            
        input_moving  = torch.from_numpy(X_vol).to(device).float()
        input_moving  = input_moving.permute(0, 4, 1, 2, 3)
        
        _, flow = model(input_moving, input_fixed)

        ############### For all anatomical structures except CSF
        # Warp segment using flow
        moving_seg = torch.from_numpy(X_seg35).to(device).float()
        moving_seg = moving_seg.permute(0, 4, 1, 2, 3)
        warp_seg   = trf(moving_seg, flow).detach().cpu().numpy()
        
        warp_seg = warp_seg.squeeze(0).squeeze(0)            
        warp_seg_new_label,good_labels = relabel(warp_seg)
        vals, labels = dice(warp_seg_new_label, atlas_seg, labels=good_labels, nargout=2)
        
        
        ############### For CSF
        moving_seg4 = torch.from_numpy(X_seg4).to(device).float()
        moving_seg4 = moving_seg4.permute(0, 4, 1, 2, 3)
        warp_seg4  = trf(moving_seg4, flow).detach().cpu().numpy()
        
        warp_seg4 = warp_seg4.squeeze(0).squeeze(0)
        warp_seg_new_label4,CSF_label = relabel_CSF(warp_seg4)
        val_CSF, _ = dice(warp_seg_new_label4, atlas_seg, labels=CSF_label, nargout=2)
                
        vals_all = np.concatenate((vals, val_CSF))

        if k==0:
            dice_array = vals_all
            mean_dice = np.mean(vals_all)
        else:
            dice_array  = np.vstack((dice_array,vals_all))
            mean_dice = np.vstack((mean_dice,np.mean(vals_all) ))
    
    return dice_array, np.mean(mean_dice)



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--atlas_file",type=str,dest="atlas_file",default='../../../data/atlas_norm.npz')    
    parser.add_argument("--model_load",type=str,dest="model_load",help="Model path to be loaded")
    parser.add_argument("--reduce",type=int,dest="reduc_size",help="Reduce size of volumes")     
    parser.add_argument("--model",type=str,dest="model",choices=['vm1', 'vm2'],default='vm2',help="voxelmorph 1 or 2")
    parser.add_argument("--test_file",type=str,dest="test_file",default='../../src/test_volnames.txt',help="Test Text File")
    parser.add_argument("--slurm_data_path",type=str,dest="slurm_data_path") 


    dice_array, mean_dice =  test(**vars(parser.parse_args()))
    print(mean_dice)

