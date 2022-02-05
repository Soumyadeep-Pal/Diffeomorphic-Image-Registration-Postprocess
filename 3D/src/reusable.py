import os
import json
from pathlib import Path
import torch

def make_exp_dir(par_dir_path):
    i = 1
    if not os.path.exists(str(par_dir_path) + "/Trials/Trial %s" % i):
        pathname = str(par_dir_path) + "/Trials/Trial " + str(i)
    else:
        while os.path.exists(str(par_dir_path) + "/Trials/Trial %s" % i):
            i += 1
        pathname = str(par_dir_path) + "/Trials/Trial " + str(i)
    
    Path(pathname).mkdir()
    return pathname
      
    
def save_args(dir_path,args):
    filename = dir_path + '/Arguments_Hyperparamaters.txt'
    
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
        
def save_metrics(metric, dir_path, metric_name):
    
    if torch.is_tensor(metric):
        metric = metric.detach().cpu().numpy()
    
    file = dir_path + "/" + metric_name
    if not os.path.exists(file):
        f = open(file, "x")
    f = open(file, "a")
    f.write("%s\n" % metric)
    f.close()
    
    