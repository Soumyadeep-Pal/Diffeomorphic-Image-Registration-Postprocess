import torch
import numpy as np
from model import SpatialTransformer
import torch.nn.functional as F
from utils import get_names,name2vol
import math
import os


gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = "cuda"

def create_dst_matrix(N):
        
    size = [N,N]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors) 
    grid  = torch.stack(grids) # y, x, z
    grid = grid.type(torch.FloatTensor)
    grid = grid.to(device) 
    dst_matrix = torch.sin((grid[0,:,:]+1)*(grid[1,:,:]+1)*(math.pi/(N+1)))
    return dst_matrix

def dst_3D(X,dst_mat_D,dst_mat_H,dst_mat_W):
    
    X = X.squeeze(0)
    # import pdb;pdb.set_trace()
    dst_axis_0 = torch.einsum('ijk,il->ljk', X, dst_mat_D)
    dst_axis_1 = torch.einsum('ijk,jl->ilk', dst_axis_0, dst_mat_H)
    dst_axis_2 = torch.einsum('ijk,lk->ijl', dst_axis_1, dst_mat_W)
    
    return dst_axis_2


def idst_3D(X,dst_mat_D,dst_mat_H,dst_mat_W):
    
    D,H,W = X.shape
    dst_axis_2 = torch.einsum('ijk,lk->ijl', X, dst_mat_W)*(2/(W+1))
    dst_axis_1 = torch.einsum('ijk,jl->ilk', dst_axis_2, dst_mat_H)*(2/(H+1))
    dst_axis_0 = torch.einsum('ijk,il->ljk', dst_axis_1, dst_mat_D)*(2/(D+1))
    
    return dst_axis_0
    
    

def Grid2Jac(X,B,C,D,H,W):
    ## Let the grid be phi
    phi_X = X[:,2,:,:,:] ## IS x,y,z ok ?
    phi_Y = X[:,1,:,:,:]
    phi_Z = X[:,0,:,:,:]
        
    phi_X_dx = torch.zeros(phi_X.shape).to(device)
    phi_X_dx[:,:,:,0:W-1] = phi_X[:,:,:,1:W] - phi_X[:,:,:,0:W-1]
    phi_X_dy = torch.zeros(phi_X.shape).to(device)
    phi_X_dy[:,:,0:H-1,:] = phi_X[:,:,1:H,:] - phi_X[:,:,0:H-1,:]
    phi_X_dz = torch.zeros(phi_X.shape).to(device)
    phi_X_dz[:,0:D-1,:,:] = phi_X[:,1:D,:,:] - phi_X[:,0:D-1,:,:]
    
    
    
    phi_Y_dx = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dx[:,:,:,0:W-1] = phi_Y[:,:,:,1:W] - phi_Y[:,:,:,0:W-1]
    phi_Y_dy = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dy[:,:,0:H-1,:] = phi_Y[:,:,1:H,:] - phi_Y[:,:,0:H-1,:]
    phi_Y_dz = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dz[:,0:D-1,:,:] = phi_Y[:,1:D,:,:] - phi_Y[:,0:D-1,:,:]
    
    
    
    phi_Z_dx = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dx[:,:,:,0:W-1] = phi_Z[:,:,:,1:W] - phi_Z[:,:,:,0:W-1]
    phi_Z_dy = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dy[:,:,0:H-1,:] = phi_Z[:,:,1:H,:] - phi_Z[:,:,0:H-1,:]
    phi_Z_dz = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dz[:,0:D-1,:,:] = phi_Z[:,1:D,:,:] - phi_Z[:,0:D-1,:,:]
    
    
    return phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz

def Jac_reshape_img2mat(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B,C,D,H,W):
    
    num_pixels = D*H*W
    
    Jac = torch.zeros(B,num_pixels, C,C).to(device)
    
    Jac[:,:,0,0] = phi_X_dx.reshape(B,num_pixels)
    Jac[:,:,0,1] = phi_X_dy.reshape(B,num_pixels)
    Jac[:,:,0,2] = phi_X_dz.reshape(B,num_pixels)
    
    Jac[:,:,1,0] = phi_Y_dx.reshape(B,num_pixels)
    Jac[:,:,1,1] = phi_Y_dy.reshape(B,num_pixels)
    Jac[:,:,1,2] = phi_Y_dz.reshape(B,num_pixels)
    
    Jac[:,:,2,0] = phi_Z_dx.reshape(B,num_pixels)
    Jac[:,:,2,1] = phi_Z_dy.reshape(B,num_pixels)
    Jac[:,:,2,2] = phi_Z_dz.reshape(B,num_pixels)
    
    return Jac

def Jac_reshapre_mat2img(Jac,B,C,D,H,W):
    
    phi_X_dx = Jac[:,:,0,0].reshape(B,D,H,W)
    phi_X_dy = Jac[:,:,0,1].reshape(B,D,H,W)
    phi_X_dz = Jac[:,:,0,2].reshape(B,D,H,W)
    
    phi_Y_dx = Jac[:,:,1,0].reshape(B,D,H,W)
    phi_Y_dy = Jac[:,:,1,1].reshape(B,D,H,W)
    phi_Y_dz = Jac[:,:,1,2].reshape(B,D,H,W)
    
    phi_Z_dx = Jac[:,:,2,0].reshape(B,D,H,W)
    phi_Z_dy = Jac[:,:,2,1].reshape(B,D,H,W)
    phi_Z_dz = Jac[:,:,2,2].reshape(B,D,H,W)
    
    return phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz
    


def Jac2Lap(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B,C,D,H,W):
    
    phi_X_dx2 = torch.zeros(phi_X_dx.shape).to(device)
    phi_X_dx2[:,1:D-1,1:H-1,1:W-1] = phi_X_dx[:,1:D-1,1:H-1,1:W-1] - phi_X_dx[:,1:D-1,1:H-1,0:W-2]
    phi_X_dy2 = torch.zeros(phi_X_dy.shape).to(device)
    phi_X_dy2[:,1:D-1,1:H-1,1:W-1] = phi_X_dy[:,1:D-1,1:H-1,1:W-1] - phi_X_dy[:,1:D-1,0:H-2,1:W-1]
    phi_X_dz2 = torch.zeros(phi_X_dz.shape).to(device)
    phi_X_dz2[:,1:D-1,1:H-1,1:W-1] = phi_X_dz[:,1:D-1,1:H-1,1:W-1] - phi_X_dz[:,0:D-2,1:H-1,1:W-1]
    
    Lap_X = phi_X_dx2 + phi_X_dy2 + phi_X_dz2
    
    
    phi_Y_dx2 = torch.zeros(phi_Y_dx.shape).to(device)
    phi_Y_dx2[:,1:D-1,1:H-1,1:W-1] = phi_Y_dx[:,1:D-1,1:H-1,1:W-1] - phi_Y_dx[:,1:D-1,1:H-1,0:W-2]
    phi_Y_dy2 = torch.zeros(phi_Y_dy.shape).to(device)
    phi_Y_dy2[:,1:D-1,1:H-1,1:W-1] = phi_Y_dy[:,1:D-1,1:H-1,1:W-1] - phi_Y_dy[:,1:D-1,0:H-2,1:W-1]
    phi_Y_dz2 = torch.zeros(phi_Y_dz.shape).to(device)
    phi_Y_dz2[:,1:D-1,1:H-1,1:W-1] = phi_Y_dz[:,1:D-1,1:H-1,1:W-1] - phi_Y_dz[:,0:D-2,1:H-1,1:W-1]
    
    Lap_Y = phi_Y_dx2 + phi_Y_dy2 + phi_Y_dz2
    
    
    phi_Z_dx2 = torch.zeros(phi_Z_dx.shape).to(device)
    phi_Z_dx2[:,1:D-1,1:H-1,1:W-1] = phi_Z_dx[:,1:D-1,1:H-1,1:W-1] - phi_Z_dx[:,1:D-1,1:H-1,0:W-2]
    phi_Z_dy2 = torch.zeros(phi_Z_dy.shape).to(device)
    phi_Z_dy2[:,1:D-1,1:H-1,1:W-1] = phi_Z_dy[:,1:D-1,1:H-1,1:W-1] - phi_Z_dy[:,1:D-1,0:H-2,1:W-1]
    phi_Z_dz2 = torch.zeros(phi_Z_dz.shape).to(device)
    phi_Z_dz2[:,1:D-1,1:H-1,1:W-1] = phi_Z_dz[:,1:D-1,1:H-1,1:W-1] - phi_Z_dz[:,0:D-2,1:H-1,1:W-1]
    
    Lap_Z = phi_Z_dx2 + phi_Z_dy2 + phi_Z_dz2
    
    
    
    return Lap_X, Lap_Y, Lap_Z
    

def MatrixExp(X, full=False): 
    # X shape: num_pixels x 3 x 3
    A = torch.eye(3).repeat(X.shape[0], X.shape[1], 1, 1).to(device)
    
    if full==True:
        Mexp = torch.eye(3).repeat(X.shape[0], X.shape[1], 1, 1).to(device)
        for i in torch.arange(1,20):
           A = torch.matmul(A/i,X)  
           Mexp = Mexp + A
    else:
        Mexp = torch.zeros(3).repeat(X.shape[0], X.shape[1], 1, 1).to(device)
        A = torch.matmul(A/1,X)
        for i in torch.arange(2,20):
           A = torch.matmul(A/i,X)  
           Mexp = Mexp + A
   
    return Mexp

def create_grid(D,H,W):
    
    size = [D,H,W]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors) 
    grid  = torch.stack(grids) # y, x, z
    grid  = torch.unsqueeze(grid, 0)  #add batch
    grid = grid.type(torch.FloatTensor)
    grid = grid.to(device)
    return grid
    
    
    
def plugin_dst(flow_init,dst_mat_D,dst_mat_H,dst_mat_W,eig):
    
    B,C,D,H,W = flow_init.shape
    phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz = Grid2Jac(flow_init,B,C,D,H,W)
    Jac = Jac_reshape_img2mat(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B,C,D,H,W)
    Jac_exp = MatrixExp(Jac, full=True)
    phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz = Jac_reshapre_mat2img(Jac_exp,B,C,D,H,W)
    Lap_X, Lap_Y, Lap_Z = Jac2Lap(phi_X_dx, phi_X_dy, phi_X_dz, phi_Y_dx, phi_Y_dy, phi_Y_dz, phi_Z_dx, phi_Z_dy, phi_Z_dz, B,C,D,H,W)
    
    Lap_X_in = Lap_X[:,1:D-1,1:H-1,1:W-1]
    
    Lap_X_dst = dst_3D(Lap_X_in,dst_mat_D,dst_mat_H,dst_mat_W)/eig
    Lap_X_iden = idst_3D(Lap_X_dst,dst_mat_D,dst_mat_H,dst_mat_W)
    flow_rec_x = Lap_X_iden.unsqueeze(0)
    
    
    Lap_Y_in = Lap_Y[:,1:D-1,1:H-1,1:W-1]
    Lap_Y_dst = dst_3D(Lap_Y_in,dst_mat_D,dst_mat_H,dst_mat_W)/eig
    Lap_Y_iden = idst_3D(Lap_Y_dst,dst_mat_D,dst_mat_H,dst_mat_W)
    flow_rec_y = Lap_Y_iden.unsqueeze(0)
    
    
    Lap_Z_in = Lap_Z[:,1:D-1,1:H-1,1:W-1]
    Lap_Z_dst = dst_3D(Lap_Z_in,dst_mat_D,dst_mat_H,dst_mat_W)/eig
    Lap_Z_iden = idst_3D(Lap_Z_dst,dst_mat_D,dst_mat_H,dst_mat_W)
    flow_rec_z = Lap_Z_iden.unsqueeze(0)
    
    flow_rec = torch.stack([flow_rec_z, flow_rec_y, flow_rec_x], dim = 1)
        
    flow_final = torch.zeros(flow_init.shape).to(device)
    flow_final[:,:,1:D-1,1:H-1,1:W-1] = flow_rec
    
    
    phi_X_dx_f, phi_X_dy_f, phi_X_dz_f, phi_Y_dx_f, phi_Y_dy_f, phi_Y_dz_f, phi_Z_dx_f, phi_Z_dy_f, phi_Z_dz_f = Grid2Jac(flow_final,B,C,D,H,W)
    Jac_final = Jac_reshape_img2mat(phi_X_dx_f, phi_X_dy_f, phi_X_dz_f, phi_Y_dx_f, phi_Y_dy_f, phi_Y_dz_f, phi_Z_dx_f, phi_Z_dy_f, phi_Z_dz_f, B,C,D,H,W)
    
        
    return flow_final, Jac_exp, Jac_final
    
    
        

