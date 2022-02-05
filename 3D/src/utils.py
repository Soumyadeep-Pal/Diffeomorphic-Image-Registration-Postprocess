import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from ast import literal_eval
import datagenerators
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.transform import pyramid_gaussian



def get_names(filename,slurm_data_path, seg=False):
    
    file = open(filename)
    strings = file.readlines()
    strings = [x.strip() for x in strings]
    strings = literal_eval(strings[0])
    vol_names = [slurm_data_path + name + '/aligned_norm.nii.gz' for name in strings]
    if seg==True:
        seg35_names = [slurm_data_path + name + '/aligned_seg35.nii.gz' for name in strings]
        seg4_names = [slurm_data_path + name + '/aligned_seg4.nii.gz' for name in strings]
        return vol_names, seg35_names, seg4_names
    return vol_names

def name2vol(vol_name, reduc_size, seg=False):
    
    X_vol = datagenerators.load_volfile(vol_name)
    X_vol = X_vol[np.newaxis, ..., np.newaxis]
    if reduc_size == 1:
        if seg == False:
            vol = resize(X_vol, (1,160/2,192/2,224/2,1))
        else:
            vol = np.round(resize(X_vol, (1,160/2,192/2,224/2,1)))
    elif reduc_size == 0:
        if seg == False:
            vol = X_vol
        else:
            vol = np.round(X_vol)
    return vol

def relabel(warp_seg):
    
    warp_seg_new_label = np.zeros(warp_seg.shape)
    neuritelabel=[13,7,26,6,25,1,20,5,24,9,28,17,33,10,29,8,27,3,22,14,30,11,12,15,31,2,21,19,35]
    # freesurferlabel = [16,9,48,8,47,2,41,7,46,12,51,28,60,13,52,11,50,4,43,17,53,14,15,18,54,3,42,31,63]
    freesurferlabel = [16,10,49,8,47,2,41,7,46,12,51,28,60,13,52,11,50,4,43,17,53,14,15,18,54,3,42,31,63]
    
    for i in range(len(neuritelabel)):
        warp_seg_new_label[warp_seg==neuritelabel[i]]=freesurferlabel[i]
    
    return warp_seg_new_label, freesurferlabel

def relabel_CSF(atlas_seg):
    
    atlas_seg_new_label = np.zeros(atlas_seg.shape)
    neuritelabel=[4,4,4,4]
    freesurferlabel = [4,14,15,43]
    
    for i in range(len(neuritelabel)):
        atlas_seg_new_label[atlas_seg==freesurferlabel[i]]=neuritelabel[i]
    
    return atlas_seg_new_label,[neuritelabel[0]]
    

def make_grid(inp):
    
    B,C,D,H,W = inp.shape   
    size = [D,H,W]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors) 
    grid  = torch.stack(grids) # y, x, z
    grid  = torch.unsqueeze(grid, 0)  #add batch
    grid = grid.type(torch.FloatTensor)
    grid = grid.to(device)
    
    return grid
    
def make_pyramid(image, downscale, levels, grid=True):
    
    pyramid = tuple(pyramid_gaussian(gaussian(image,1.0,multichannel=False), downscale=downscale, multichannel=False))
    
    image_pyramid = []
    grid_pyramid = []
    
    for s in range(levels):
      image_tensor = torch.from_numpy(pyramid[s]).to(device).float()
      image_tensor  = image_tensor.permute(0, 4, 1, 2, 3)
      image_pyramid.append(image_tensor)
      
      if grid==True:
          grid_tensor = make_grid(image_tensor)
          grid_pyramid.append(grid_tensor)
    
  
    if grid==True:
        return image_pyramid,grid_pyramid
    else:
        return image_pyramid
      
      
      
      
      
      
      
  