# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:06:00 2022

@author: ASUS
"""

import torch
import os
from os import path
import torch.nn as nn
import copy
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from torchvision import models,transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import argparse
import time
import presets
import numpy as np
from typing import List, Optional, Tuple

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--maxEpochs', default =500, type = int)
parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-5)
parser.add_argument('--norm_weight_decay', type=float, default=0.0)
parser.add_argument('--res_dir',type=str,default="resnet_cls")
args = parser.parse_args()
torch.manual_seed(args.seed)
if(os.path.isdir(args.res_dir)==False ):
  os.mkdir(args.res_dir) 
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class WheatDataset(Dataset):
    """wheat dataset."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.imgs=[]
        self.targets=[]
        files=os.listdir(self.root_dir)
        for f in files:
         file_path=path.join(self.root_dir, f)
         if path.isfile(file_path) and file_path.endswith(".png"):
             img=pil_loader(file_path)
             img=img.resize(size=(232,232),resample=Image.BICUBIC)
             self.imgs.append(img)
             info=f.split(sep='~')[1]
             value=np.float32(info[:-4])
             if value<=4.0:
                 self.targets.append(0)
             elif value<=8.0:
                 self.targets.append(1)
             else:
                  self.targets.append(2)                  

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.imgs[idx])
        return image,self.targets[idx]


def kfold_split(X, n_splits):
    indices = np.arange(len(X))
    np.random.shuffle(indices)  #
    fold_size = len(X) // n_splits  #
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end1 = (i + 1) * fold_size
        end2 = (i + 2) * fold_size
        if i == n_splits - 1:  #
            end1 = len(X)
            end2 = fold_size
        if i == n_splits - 2:
            end2 = end2 = len(X)
        if i < n_splits -1:
            test_indices = indices[start:end1]
            validation_indices = indices[end1:end2]
            train_indices = np.append(copy.deepcopy(indices[:end1]), copy.deepcopy(indices[end2:]))
        else:
            test_indices = indices[start:end1]
            validation_indices = indices[0:end2]
            train_indices = indices[end2:start]
        folds.append((i, (train_indices, validation_indices, test_indices)))
    return folds

def train(model,dataloader,optimizer,criterion,device):
    model.train()
    total_loss = 0
    count = 0
    for i, (inputs, targets) in enumerate(dataloader):        
        inputs, targets = inputs.to(device), targets.to(device)     
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        count+=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/count

def eval(model,dataloader,device):
    model.eval()    
    correct_num = 0.0
    total_sample = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)        
            out = model(inputs)
            preds = torch.argmax(out, dim=1)
            correct_num +=(preds == targets).float().sum()           
            total_sample+=inputs.shape[0]
    return 100*correct_num/total_sample

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups

use_cuda=torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(232),    
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])       
    ]),
    'val': transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
    ]),
}


train_crop_size=176
train_transformer=presets.ClassificationPresetTrain(
                crop_size=train_crop_size,                
                auto_augment_policy='ta_wide',
                random_erase_prob=0.1,
            )    
    

def kfold_split_3(CK_0, CK_1, CK_2, n_splits):
    ckl0 = len(CK_0)
    ckl1 = len(CK_1)
    ckl2 = len(CK_2)
    ck0 = kfold_split(CK_0, n_splits)
    ck1 = kfold_split(CK_1, n_splits)
    ck2 = kfold_split(CK_2, n_splits)
    folds = []
    for (i, (t_idx_ck0, val_idx_ck0, test_idx_ck0)), (j, (t_idx_ck1, val_idx_ck1, test_idx_ck1)), (k, (t_idx_ck2, val_idx_ck2, test_idx_ck2))in zip(ck0, ck1 ,ck2):
        t_idx_ck1 = t_idx_ck1 + ckl0
        t_idx_ck2 = t_idx_ck2 + ckl0 + ckl1
        t_idx_ck = np.concatenate((t_idx_ck0, t_idx_ck1, t_idx_ck2), axis=0)
        val_idx_ck1 = val_idx_ck1 + ckl0
        val_idx_ck2 = val_idx_ck2 + ckl0 + ckl1
        val_idx_ck = np.concatenate((val_idx_ck0, val_idx_ck1, val_idx_ck2), axis=0)
        test_idx_ck1 = test_idx_ck1 + ckl0
        test_idx_ck2 = test_idx_ck2 + ckl0 + ckl1
        test_idx_ck = np.concatenate((test_idx_ck0, test_idx_ck1, test_idx_ck2), axis=0)
        folds.append((i, (t_idx_ck, val_idx_ck, test_idx_ck)))
    return folds

D0 = WheatDataset(root_dir='data/data/D0', transform=train_transformer)
D1 = WheatDataset(root_dir='data/data/D1', transform=train_transformer)
D2 = WheatDataset(root_dir='data/data/D2', transform=train_transformer)




dataset = ConcatDataset([D0, D1, D2])
print(len(dataset))
resnet50= models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_feats = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_feats, 3)
model=resnet50.to(device)
 

criterion = nn.CrossEntropyLoss(label_smoothing=0.05) 


param_groups=set_weight_decay(model,args.weight_decay,args.norm_weight_decay)
optimizer = torch.optim.Adam(param_groups,lr=args.lr,amsgrad=False)


kf = kfold_split_3(D0, D1, D2, 10)
test_acc = []
print(len(dataset))
for (epoch, (train_idx, val_idx, test_idx)) in kf:
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=128, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler, num_workers=0)
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, device)
    print("{}th Epoch took {:.3f}s".format((epoch + 1), time.time() - start_time))
    print("  training loss:\t\t{:.3f}".format(train_loss))
    eval_accu = eval(model, val_loader, device)
    print("  validation accuracy:\t\t{:.2f}".format(eval_accu))
    torch.save(model.state_dict(), args.res_dir + '/{}d_best_model.pth'.format(epoch))
    model.load_state_dict(torch.load(args.res_dir + '/{}d_best_model.pth'.format(epoch)))
    test_accu = eval(model, test_loader, device)
    test_acc.append(test_accu.cpu())
    print("  test accuracy:\t\t{:.2f}".format(eval_accu))
avg_acc = np.mean(test_acc)
print("avg  test accuracy:\t\t{:.2f}".format(avg_acc))
