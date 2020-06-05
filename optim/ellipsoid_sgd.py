import torch
from torch.optim.optimizer import Optimizer, required
DEBUG=1


class Ellipsoid_SGD(Optimizer):
    r"""Implements Ellipsoid method (for reference chapter 2 https://arxiv.org/pdf/1405.4980.pdf) under stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of ellipsoid method under SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
        after that gradient g is using in ellipsoid method:
        math:: 
        \begin{algorithm}
            \caption{Ellipsoid method for minimization of function $v(y)$}
	        \label{alg:ellipsoid}
	        \begin{algorithmic}[1]
                \REQUIRE Number of iterations $N \geq 1$, sphere $\mB_{\mR} 	\supseteq Q_y$, its center $c$ and radius $\mR$.
                \STATE $\mE_0 := \mB_{\mR},\quad H_0 := \mR^2 I_n,\quad c_0 := c$.
                \FOR{$k=0,\, \dots, \, N-1$.}
                    \IF {$c_k \in Q_y$}
                        \STATE $g_k := g \in \partial_{\delta} v(c_k)$
                        \IF {$g_k = 0$}
                            \STATE $\tilde{y} := c_k$
                            \RETURN $\tilde{y}$
                        \ENDIF
                    \ELSE
                        \STATE $g_k := g$, where $g \neq 0$ such that $Q_y \subset \{ y \in \mE_k: g^T (y-c_k) \leq 0 \}$
                    \ENDIF
                    \STATE $c_{k+1} := c_k - \ddfrac{1}{n+1}\ddfrac{H_k g_k}{\sqrt{g_k^T H_k g_k}}$ \\
                    $H_{k+1} := \ddfrac{n^2}{n^2-1} \left( H_k - \ddfrac{2}{n+1}\ddfrac{H_k g_k g_k^T H_k}{g_k^T H_k g_k} \right)$ \\
                    $\mE_{k+1} := \{y: (y-c_{k+1})^T H_{k+1}^{-1} (y-c_{k+1}) \leq 1 \}$
                \ENDFOR
                \ENSURE $y^N = \arg\min\limits_{y \in \{c_0, ..., c_N \} \cap Q_y } v(y)$
	        \end{algorithmic}
        \end{algorithm}
    """

    def __init__(self, params, H, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
#         if H is none:
#             raise ValueError("Invalid H: {}".format(H))
        self.H = H
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Ellipsoid_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ellipsoid_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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

        for group, H_group in zip(self.param_groups, self.H):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, H in zip(group['params'], H_group):
                if p.grad is None:
                    continue
                n = H.size()[0]
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if DEBUG>0:
                    print("parameter shape: ", p.size(),
                    "gradient shape: ", d_p.size(),
                    "Ellipsoid matrix shape: ", H.size())
                d_p_shape = d_p.shape
                d_p = d_p.reshape(1,-1)
                denom_H = d_p.matmul(H).matmul(d_p.t())
                p.add_(H.matmul(d_p).div(denom_H.sqrt()), alpha=-1/(n+1)*group['lr'])
                H.add_((H.matmul(d_p).matmul(d_p.t()).matmul(H)).div(denom_H), alpha=-2/(n+1))
                H.mul_(n**2/(n**2-1))
        return loss