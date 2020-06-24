import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import copy
import argparse
from layers.modules import MultiBoxLoss

class VAT(nn.Module):
    def __init__(self, eps=1.0, xi=1e-6, n_iteration=1):   
        super().__init__()
        self.eps = eps
        self.xi = xi
        self.n_iteration = n_iteration

    def normalize(self, v):               
        v = v / (1e-12 + self.__reduce_max(v.abs(), range(1, len(v.shape))))
        v = v / (1e-6 + v.pow(2).sum((1,2,3),keepdim=True)).sqrt()
        return v

    def kld(self, q_logit, p_logit):
        q = q_logit.softmax(1)
        qlogp = (q * self.__logsoftmax(p_logit)).sum(1)
        qlogq = (q * self.__logsoftmax(q_logit)).sum(1)
        return qlogq - qlogp

    def forward(self, x, y, semis, net, iteration):   								  
        d = torch.randn_like(x)    					  
        d = self.normalize(d)
        x = x.clone()
        y = copy.deepcopy(y)
        semis = copy.deepcopy(semis) 
        
        for _ in range(self.n_iteration):
            d.requires_grad = True
            x_hat = x + self.xi * d
            out, conf, conf_flip, loc, loc_flip = net(x, x_hat, isflip=True)   

            criterion = MultiBoxLoss(21 , 0.5, True, 0, True, 3, 0.5, False, True) 
            conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

            sup_image_binary_index = np.zeros([len(semis),1])

            for super_image in range(len(semis)):
                if(int(semis[super_image])==1):
                    sup_image_binary_index[super_image] = 1
                else:
                    sup_image_binary_index[super_image] = 0
                
                if(int(semis[len(semis)-1-super_image])==0):
                    del y[len(semis)-1-super_image]

            sup_image_index = np.where(sup_image_binary_index == 1)[0]
            unsup_image_index = np.where(sup_image_binary_index == 0)[0]

            loc_data, conf_data, priors = out           

            if (len(sup_image_index) != 0):
                loc_data = loc_data[sup_image_index,:,:]
                conf_data = conf_data[sup_image_index,:,:]
                output = (
                    loc_data,
                    conf_data,
                    priors
                )

            loss_l = Variable(torch.cuda.FloatTensor([0]))
            loss_c = Variable(torch.cuda.FloatTensor([0]))
            
            sampling = True
            if(sampling is True):
                conf_class = conf[:,:,1:].clone()
                background_score = conf[:, :, 0].clone()
                each_val, each_index = torch.max(conf_class, dim=2)
                mask_val = each_val > background_score	
                mask_val = mask_val.data

                mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
                mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

                conf_mask_sample = conf.clone()
                loc_mask_sample = loc.clone()
                conf_sampled = conf_mask_sample[mask_conf_index].view(-1, 21)
                loc_sampled = loc_mask_sample[mask_loc_index].view(-1, 4)

                conf_mask_sample_flip = conf_flip.clone()
                loc_mask_sample_flip = loc_flip.clone()
                conf_sampled_flip = conf_mask_sample_flip[mask_conf_index].view(-1, 21)
                loc_sampled_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)   


            if(mask_val.sum()>0):
                ## CLS LOSS 
                conf_sampled_flip = conf_sampled_flip + 1e-7
                conf_sampled = conf_sampled + 1e-7
                consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(), conf_sampled_flip.detach()).sum(-1).mean()  
                consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(), conf_sampled.detach()).sum(-1).mean()  
                consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b   

                ## LOC LOSS  
                consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
                consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
                consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
                consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

                consistency_loc_loss = torch.div(
                    consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                    4)  

            else:   
                consistency_conf_loss = Variable(torch.cuda.FloatTensor([0])) 
                consistency_loc_loss = Variable(torch.cuda.FloatTensor([0])) 

            consistency_loss = torch.div(consistency_conf_loss,2) + consistency_loc_loss    

            loss = consistency_loss 

            if not (mask_val.sum()>0):
                x_hat = x
                print('wrong')
            else:
                d = torch.autograd.grad(loss, d)[0] 
                d = self.normalize(d).detach()
                x_hat = x + self.eps * d
				
        return x_hat
		
    def __reduce_max(self, v, idx_list):
        for i in idx_list:
            v = v.max(i, keepdim=True)[0]
        return v

    def __logsoftmax(self,x):
        xdev = x - x.max(1, keepdim=True)[0]
        lsm = xdev - xdev.exp().sum(1, keepdim=True).log()
        return lsm
