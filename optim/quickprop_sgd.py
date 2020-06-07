#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.optim.optimizer import Optimizer, required

class QuickPropSGD(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, eps=1e-7):
      
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
            
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
            
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps)
       
        super(QuickPropSGD, self).__init__(params, defaults)
            
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=weight_decay)
                    
                if len(state) == 0 :
                    state['step'] = 0
                    state['deltap'] = p.data.clone()
                    state['gradprev'] = grad.clone()
                    p.data -= group['lr']*grad
                    state['deltap'] = p.data - state['deltap']
                    state['step'] += 1
                    continue
                  
                if state['step'] < 3 :
                    p.data -= group['lr']*grad
                    state['deltap'] = state['deltap'] * (grad / (state['gradprev'] - grad + group['eps']) )
                    state['gradprev'] = grad.clone()
                    state['step'] += 1
                    continue
                
                state['step'] += 1
                state['deltap'] *= grad / (state['gradprev'] - grad + group['eps']) 
                state['gradprev'] = grad.clone()
                p.data += group['lr'] * state['deltap']

        return loss