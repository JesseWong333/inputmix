# -*- coding: utf-8 -*-
# Author: Junjie Wang
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import numpy as np

def one_hot(x, num_classes, on_value=1., off_value=0., device='cpu'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

class InputMix:
    def __init__(self, num_classes, p, lam):
        self.num_classes = num_classes
        self.p = p
        self.lam = lam
        
    def __call__(self, inputs, targets):

        assert len(inputs) == len(self.lam), "inputs lens does not match!"
        batch_size = len(inputs[0])
        n = int(np.round(batch_size * self.p))

        changed_batches = [ modality[:n, :, :, :] for modality in inputs]
        changed_targets = [ one_hot(targets[:n], self.num_classes).clone() for i in range(len(inputs))]
       
        unchanged_batches = [ modality[n:, :, :, :] for modality in inputs]
        unchanged_target = one_hot(targets[n:], self.num_classes)
        # generate randperm
        while True:
            rand_indx_l = []
            for i in range(len(inputs)):
                rand_indx_l.append(torch.randperm(n))
            
            permutation = sum( [(rand_indx_l[i] - rand_indx_l[0]).abs() for i in range(1, len(rand_indx_l))] )
            if torch.any(permutation == 0):
                continue
            else:
                break
        
        permutated_batches = []
        permutated_t = []
        for batch_data, t, rand_indx in zip(changed_batches, changed_targets, rand_indx_l):
            batch_data = batch_data[rand_indx]
            t = t[rand_indx]
            permutated_batches.append(batch_data)
            permutated_t.append(t)
        
        weighted_target = sum([ t*lam for t, lam in zip(permutated_t, self.lam)])

        out_modalities = [ torch.cat((b1, b2), dim=0) for b1, b2 in zip(permutated_batches, unchanged_batches)]
        out_target =  torch.cat([weighted_target, unchanged_target], dim=0)
        
        return out_modalities, out_target

