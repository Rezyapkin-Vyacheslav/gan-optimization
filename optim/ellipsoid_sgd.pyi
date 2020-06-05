from typing import Iterable, Union, Callable, Optional, List
from pytorch.optim.optimizer import _params_t, Optimizer


class Ellipsoid_SGD(Optimizer):
    def __init__(self, params: _params_t, H: Iterable[Tensor], lr: float, momentum: float=..., dampening: float=..., weight_decay:float=..., nesterov:bool=...) -> None: ...
