# python imports
import os
import glob
import random
import warnings
import sys
from argparse import ArgumentParser
import git

# external imports
import numpy as np
import torch
from torch.optim import Adam
from skimage.transform import resize

sys.path.append('../../ext/medipy-lib')
sys.path.append('../../src')


# internal imports
from model import cvpr2018_net,SpatialTransformer
import datagenerators
import losses
import math

from test import test

from utils import get_names
from reusable import make_exp_dir, save_args, save_metrics



def train(dir_path,
          gpu,
          atlas_file,
          lr,
          n_iter,
          data_loss,
          model_type,
          reg_param, 
          batch_size,
          reduc_size,
          sha_ref,
          slurm_data_path):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
    if reduc_size == 1:
        atlas_vol = resize(atlas_vol, (1,160/2,192/2,224/2,1))
    
    atlas_seg = atlas['seg']
    if reduc_size == 1:
        atlas_seg = np.round(resize(atlas_seg, (160/2,192/2,224/2)))
    
    
    vol_size = atlas_vol.shape[1:-1]

    train_vol_names = get_names('../../src/training_volnames.txt',slurm_data_path)    
    

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model_type == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model_type == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")

    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)

    # Set optimizer and losses
    optimizer = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size)
    # set up atlas tensor
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    
    
    input_fixed  = torch.from_numpy(atlas_vol_bs).to(device).float()
    input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)
    B,C,D,H,W = input_fixed.shape
    
    trf = SpatialTransformer(atlas_vol.shape[1:-1])
    trf.to(device)
    
    
    best_dice = 0

    # Training loop.
    for i in range(n_iter):

        # Generate the moving images and convert them to tensors.
        moving_image = next(train_example_gen)[0]
        if reduc_size == 1:
            moving_image = resize(moving_image, (1,160/2,192/2,224/2,1))
        
        input_moving = torch.from_numpy(moving_image).to(device).float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)
                
        # # Run the data through the model to produce warp and flow field
        _, flow = model(input_moving, input_fixed)
        warp = trf(input_moving,flow)
   
            
        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed) 
        grad_loss = grad_loss_fn(flow)        
        
        loss = recon_loss + reg_param * grad_loss 
        
        save_metrics(loss, dir_path, "Training Loss")
        save_metrics(recon_loss, dir_path, "Ncc Loss")

        # Backwards and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0: 
            _, validation_avg_dice = test(gpu, atlas_file, model,reduc_size,model_type,'../../src/validation_volnames.txt',slurm_data_path)            
            save_metrics(validation_avg_dice, dir_path, "Validation Dice Score")
            
            print('Validation Dice score:',validation_avg_dice)
            
            if validation_avg_dice > best_dice:
                best_dice = validation_avg_dice
                model_path = dir_path + "/checkpoint"
                torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, model_path)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--atlas_file",type=str,dest="atlas_file",default='../../../data/atlas_norm.npz')
    parser.add_argument("--lr", type=float,dest="lr", help="learning rate")
    parser.add_argument("--n_iter",type=int,dest="n_iter", help="number of iterations") 
    parser.add_argument("--data_loss",type=str,dest="data_loss",default='ncc',help="data_loss: mse of ncc")
    parser.add_argument("--model_type",type=str,dest="model_type",choices=['vm1', 'vm2'],default='vm2',help="voxelmorph 1 or 2")
    parser.add_argument("--lambda",type=float,dest="reg_param",help="regularization parameter")
    parser.add_argument("--batch_size",type=int,dest="batch_size",default=1,help="batch_size")
    parser.add_argument("--reduce",type=int,dest="reduc_size",help="Reduce size of volumes")   

    
    repo = git.Repo(search_parent_directories=True)
    sha_ref = repo.head.object.hexsha
    parser.add_argument("--sha_ref",dest="sha_ref",required=False,default=sha_ref)
    parser.add_argument("--slurm_data_path",type=str,dest="slurm_data_path")   
    
    args = parser.parse_args()
    
    # Make Trial Directory
    par_dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = make_exp_dir(par_dir_path)

    save_args(dir_path,args)
    
    train(dir_path,**vars(parser.parse_args()))





