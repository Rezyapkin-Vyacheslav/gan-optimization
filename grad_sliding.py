import torch
from torch.optim.optimizer import Optimizer
from math import ceil

class GradSliding(Optimizer):
    """Lan's Gradient Sliding algorithm."""
    def __init__(self, params, L, M, D_tilde):
        defaults = dict(L=L, M=M, D_tilde=D_tilde)
        super().__init__(params, defaults)
        self.k = 0
        self.t = 0
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
        # print(f">>> gamma={self.gamma:.2f}, T={self.T:.2f}, P={self.P:.2f}, beta={self.beta:.2f}")
    
    def upd_PS_parameters(self):
        """
        Update parameters of PS procedure.
        
        Increment t (counter in PS procedure). Calculate p and theta
        according to formula (8.1.39) in Lan's book. If this is the last
        PS iteration, change mode to main and reset counter.
        """
        self.t += 1
        self.p = self.t / 2
        self.theta = 2 * (self.t + 1) / (self.t * (self.t + 3))
        # print(f"p={self.p:.2f}, theta={self.theta:.2f}")
        
        if self.t % self.T == 0:
            self.t = 0
            self.mode = 'main'

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

                    # Formula (1) from our report.
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
                        x_underbar = (1 - self.gamma_next) * state['x_bar'] \
                            + self.gamma_next * state['x']
                        par.copy_(x_underbar)
                
        return loss