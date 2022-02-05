# python imports
import os
import sys
from argparse import ArgumentParser
import json
import time

import numpy as np
import torch
import matplotlib.pyplot as plt


from skimage.transform import resize
import math
import nibabel as nib

sys.path.append('../../ext/medipy-lib')
sys.path.append('../../src')

from model import cvpr2018_net, SpatialTransformer
from medipy.metrics import dice
from utils import get_names,name2vol,relabel_CSF,relabel
from plugin import plugin_dst, create_grid, create_dst_matrix,Grid2Jac,Jac_reshape_img2mat
from reusable import save_metrics


def analyse(trial_path_number,slurm_data_path):

    path = "Trials/Trial " + trial_path_number  + "/Arguments_Hyperparamaters.txt"
    with open(path, 'r') as f:
        arg = json.load(f)
    
    gpu = arg['gpu']
    atlas_file = arg['atlas_file']
    model_load = "Trials/Trial " + trial_path_number + "/checkpoint"
    reduc_size = arg['reduc_size']
    model = arg['model_type']
    test_file = '../../src/test_volnames.txt'
    
    dir_path = "Trials/Trial " + trial_path_number
    dice_array, mean_dice, std_dice, neg_det_jac, time_arr = test_all(dir_path, gpu, atlas_file, model_load, reduc_size, model,test_file, slurm_data_path)
    plot_metrics(trial_path_number)
    
    
    save_metrics(mean_dice, dir_path, "Mean Dice Score")
    save_metrics(std_dice, dir_path, "Std Dice Score")
    save_metrics(neg_det_jac, dir_path, "Negative Jac Dets for TestSets")
    
    save_metrics(time_arr, dir_path, "Time Taken")
    
    
    diceboxplot(dir_path, dice_array)
    
    
    
    
def test_all(dir_path,
             gpu, 
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
   
    

    # Test file and anatomical labels we want to evaluate
    test_vol_names, test_seg35_names, test_seg4_names = get_names(test_file,slurm_data_path,seg=True)


    good_labels=[2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,26,28,30,31,41,42,43,44,46,47,48,50,51,52,53,54,58,60,62,63]
    neg_det_frac = torch.zeros(len(test_vol_names))
    time_arr = torch.zeros(len(test_vol_names))

         
    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol).to(device).float()
    input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)
    B,C,D,H,W = input_fixed.shape

    # Use this to warp segments
    trf = SpatialTransformer(atlas_vol.shape[1:-1], mode='nearest')
    trf.to(device)
    
    grid = create_grid(D,H,W)


    for k in range(0, len(test_vol_names)):
        
        start_time = time.time()
        
        X_vol = name2vol(test_vol_names[k]  , reduc_size, seg=False)
        X_seg35 = name2vol(test_seg35_names[k], reduc_size, seg=True)
        X_seg4 = name2vol(test_seg4_names[k] , reduc_size, seg=True)
            
        input_moving  = torch.from_numpy(X_vol).to(device).float()
        input_moving  = input_moving.permute(0, 4, 1, 2, 3)
        
        _, flow = model(input_moving, input_fixed)
        
        end_time = time.time()
        time_arr[k] = end_time - start_time 

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
        atlas_seg_new_label,CSF_label = relabel_CSF(atlas_seg)
        val_CSF, _ = dice(warp_seg4, atlas_seg_new_label, labels=CSF_label, nargout=2)
                
        vals_all = np.concatenate((vals, val_CSF))
        
        
        
        save_metrics(list(vals_all), dir_path, "Dice Scores for anatomical structures2")

        if k==0:
            dice_array = vals_all
            mean_dice = np.mean(vals_all)
        else:
            dice_array  = np.vstack((dice_array,vals_all))
            mean_dice = np.vstack((mean_dice,np.mean(vals_all) ))
        
        ## Jac Det
        deformfield = grid + flow
        B,C,D,H,W = deformfield.shape 
        phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz = Grid2Jac(deformfield,B,C,D,H,W)
        Jac = Jac_reshape_img2mat(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B,C,D,H,W)
        Jac_det = torch.det(Jac)
        neg_det_frac[k] = torch.sum(Jac_det<0) / Jac_det.shape[1]
    
    return dice_array, np.mean(mean_dice), np.std(mean_dice), neg_det_frac, time_arr

def diceboxplot(dir_path, dice_array):
    
    dice = np.zeros([dice_array.shape[0],17])
    dice[:,0] = dice_array[:,0]
    dice[:,1] = (dice_array[:,1] + dice_array[:,2])/2
    dice[:,2] = (dice_array[:,3] + dice_array[:,4])/2
    dice[:,3] = (dice_array[:,5] + dice_array[:,6])/2
    dice[:,4] = (dice_array[:,7] + dice_array[:,8])/2
    dice[:,5] = (dice_array[:,9] + dice_array[:,10])/2
    dice[:,6] = (dice_array[:,11] + dice_array[:,12])/2
    dice[:,7] = (dice_array[:,13] + dice_array[:,14])/2
    dice[:,8] = (dice_array[:,15] + dice_array[:,16])/2
    dice[:,9] = (dice_array[:,17] + dice_array[:,18])/2
    dice[:,10] = (dice_array[:,19] + dice_array[:,20])/2
    dice[:,11] = dice_array[:,21]
    dice[:,12] = dice_array[:,22]
    dice[:,13] = (dice_array[:,23] + dice_array[:,24])/2
    dice[:,14] = (dice_array[:,25] + dice_array[:,26])/2
    dice[:,15] = (dice_array[:,27] + dice_array[:,28])/2
    dice[:,16] = dice_array[:,29]
    
    fig1, ax1 = plt.subplots(figsize=(30,5))
    ax1.set_title('Dice Scores')
    ax1.boxplot(dice)
    
    x = np.arange(1,18)
    values = ['Brain-Stem','Thalamus','Cerebellum-Cortex','Cerebral-WM','Cerebellum-WM','Putamen','Ventral-DC','Pallidum','Caudate','Lateral-Ventricle','Hippocampus','3rd-Ventricle','4th-Ventricle','Amygdala','Cerebral-Cortex','Choroid-Plexus','CSF']
    plt.xticks(x,values)
    
    plt.savefig(dir_path + "/BoxPlot of Dice Scores of Anatomical Structures")

def plot_metrics(trial_path_number):
    
    dir_path = "Trials/Trial " + trial_path_number
    
    with open(dir_path + "/Training Loss") as train_loss_file:
        lines = train_loss_file.readlines()
        train_loss = [float(line.rstrip()) for line in lines]
        
        
    with open(dir_path + "/Validation Dice Score") as val_dice_file:
        lines = val_dice_file.readlines()
        val_dice = [float(line.rstrip()) for line in lines]
        
    with open(dir_path + "/Ncc Loss") as ncc_loss_file:
        lines = ncc_loss_file.readlines()
        ncc_loss = [float(line.rstrip()) for line in lines]
        
    val_epoch = np.arange(1, len(val_dice)+1)*100

    fig1, ax1 = plt.subplots()
    ax1.plot(val_epoch,val_dice)

    val_dice_y_max = np.max(val_dice)
    val_dice_x_max = np.nonzero(val_dice==val_dice_y_max)[0][0]
    ax1.annotate("Max_value={:.3f}".format(val_dice_y_max), xy=(val_dice_x_max*100, val_dice_y_max), xytext=(10000,0.6),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Dice Score')
    fig1.savefig(dir_path + "/Validation_plot")


    loss_epoch = np.arange(1, len(train_loss)+1)

    fig2, ax2 = plt.subplots()
    ax2.plot(loss_epoch,train_loss)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Training Loss')
    fig2.savefig(dir_path + "/Loss_plot")


    ncc_epoch = np.arange(1, len(ncc_loss)+1)

    fig3, ax3 = plt.subplots()
    ax3.plot(ncc_epoch,ncc_loss)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('NCC Loss')
    fig3.savefig(dir_path + "/ncc_plot")
    
    
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--expnum", type=str, dest="trial_path_number")
    parser.add_argument("--slurm_data_path",type=str,dest="slurm_data_path")   
    args = parser.parse_args()
    # analyse(args.trial_path_number)
    analyse(**vars(parser.parse_args()))

