from torch.optim.optimizer import _params_t, Optimizer

class QuickPropSGD(Optimizer):
    def __init__(self, params: _params_t, lr: float, momentum: float=..., dampening: float=..., weight_decay:float=..., nesterov:bool=..., eps:float=...) -> 
