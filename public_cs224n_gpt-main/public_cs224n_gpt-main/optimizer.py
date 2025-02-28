from typing import Callable, Iterable, Tuple
import math

import torch
from sympy.abc import epsilon
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                # indexed by param (saves 1st and 2nd momments)
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE
                if not state: # for the first iteration
                    state['t'] = 0
                    state['m'] = 0
                    state['v'] = 0
                state['t'] += 1
                beta_1, beta_2 = group['betas']
                m_t = beta_1 * state['m'] + (1-beta_1) * grad
                v_t = beta_2 * state['v'] + (1-beta_2) * (grad **2)
                if group['correct_bias']:
                    alpha_t = alpha * math.sqrt(1-beta_2**state['t'])/(1-beta_1**state['t'])
                    theta_t = p - alpha_t * m_t / (v_t**0.5 + group['eps'])
                else:
                    theta_t = p - alpha * m_t / (v_t**0.5 + group['eps'])
                theta_t -= group['weight_decay'] * alpha * p
                p.data = theta_t
                state['m'] = m_t
                state['v'] = v_t
        return loss
