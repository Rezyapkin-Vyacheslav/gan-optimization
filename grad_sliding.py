import torch
from torch.optim.optimizer import Optimizer
from math import ceil
"""
Gradient Sliding algorithm.

The algorithm is proposed by G. Lan in paper:
https://arxiv.org/pdf/1406.0919.pdf
and also described in his textbook:
http://pwp.gatech.edu/guanghui-lan/wp-content/uploads/sites/330/2019/08/LectureOPTML.pdf
In this code, we reference formulas from the textbook.
"""

class GradSliding(Optimizer):
    def __init__(self, params, L, M, D_tilde):
        """
        Create optimizer with specified parameters.
        
        Meaning of parameters L, M and D_tilde is explained, respectively,
        in formulas (8.1.2), (8.1.2) and Corollary 8.2. (page 497) in
        Lan's book.
        """
        defaults = dict(L=L, M=M, D_tilde=D_tilde)
        super().__init__(params, defaults)
        self.k = 0
        self.t = 0
        self.steps = 0
        self.mode = 'main'
    
    def upd_main_parameters(self):
        """
        Update parameters of main loop of gradient sliding.
        
        Increment k (counter in main loop). Change mode to PS
        (prox-sliding procedure). Calculate parameters according to the
        formulas in Lan's book:
        gamma, T, beta - formula (8.1.42); T - formula (8.1.42).
        gamma_next is value of gamma in the next iteration.
        """
        self.steps += 1
        self.k += 1
        self.mode = 'PS'

        self.gamma = 3 / (self.k + 2)
        self.gamma_next = 3 / (self.k + 3)
        
        L = self.defaults['L']
        M = self.defaults['M']
        D_tilde = self.defaults['D_tilde']
        T = ceil(M**2 * (self.k + 1)**3 / (D_tilde * L**2))
        self.T = int(T)
        
        self.P = 2 / ((self.T + 1) * (self.T + 2))
        self.beta = 9 * L * (1 - self.P) / (2 * (self.k + 1))
    
    def upd_PS_parameters(self):
        """
        Update parameters of PS procedure.
        
        Increment t (counter in PS procedure). Calculate p and theta
        according to formula (8.1.39) in Lan's book. If this is the last
        PS iteration, change mode to main and reset counter.
        """
        self.steps += 1
        self.t += 1
        self.p = self.t / 2
        self.theta = 2 * (self.t + 1) / (self.t * (self.t + 3))

    @torch.no_grad()
    def to_eval(self):
        """Load x_bar as model parameters.
        
        Recommended to do only at the beginning of main loop."""
        for group in self.param_groups:
            for par in group['params']:
                if par.grad is None:
                    continue
                state = self.state[par]
                par.copy_(state['x_bar'])

    @torch.no_grad()
    def to_train(self):
        """Load x_underbar as model parameters.
        
        Recommended to do only at the beginning of main loop."""
        for group in self.param_groups:
            for par in group['params']:
                if par.grad is None:
                    continue
                state = self.state[par]
                par.copy_(state['x_underbar'])
        
    @torch.no_grad()
    def step(self, closure=None):
        """Perform Gradient Sliding step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Part of main loop before PS (prox-sliding) procedure.
        # In this branch, par is x_underbar in notation of Lan's book.
        if self.mode == 'main':
            self.upd_main_parameters()
            for group in self.param_groups:
                for par in group['params']:
                    if par.grad is None:
                        continue
                    
                    state = self.state[par]
                    # State initialization.
                    if len(state) == 0:
                        state['x'] = par.clone()
                        state['x_bar'] = par.clone()
                        state['x_underbar'] = par.clone()
                    
                    state['df_x'] = par.grad
                    # At the beginning of PS procedure, gradient of h
                    # will be calculated at u0 = x.
                    par.copy_(state['x'])
                
        # PS procedure.
        # In this branch, par is u in notation of Lan's book.
        elif self.mode == 'PS':
            self.upd_PS_parameters()
            for group in self.param_groups:
                for par in group['params']:
                    if par.grad is None:
                        continue
                    
                    state = self.state[par]
                    if self.t == 1:
                        state['u_tilde'] = par.clone()
                    
                    dh_u = par.grad

                    # Formula (3) from our report.
                    numerator = self.beta * (state['x'] + self.p * par) \
                              - state['df_x'] - dh_u
                    par.copy_(numerator / (self.beta * (1 + self.p)))
                    
                    state['u_tilde'] = (1 - self.theta) * state['u_tilde'] \
                                     + self.theta * par
                    
                    if self.t % self.T == 0:
                        # Finish PS procedure.
                        state['x'] = par
                        state['x_tilde'] = state['u_tilde']
                
                        # Part of main loop after PS procedure.
                        state['x_bar'] = (1 - self.gamma) * state['x_bar'] \
                                       + self.gamma * state['x_tilde']
                        # Beginning of main loop of new iteration.
                        # Now par is again x_underbar.
                        state['x_underbar'] = self.gamma_next * state['x'] \
                            + (1 - self.gamma_next) * state['x_bar']
                        par.copy_(state['x_underbar'])
            if self.t % self.T == 0:
                self.t = 0
                self.mode = 'main'
                
        return loss