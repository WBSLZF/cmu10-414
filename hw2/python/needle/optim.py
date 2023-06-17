"""Optimization module"""
from numpy import float32, float64

import python.needle as ndl
import numpy as np

from python.needle import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            #print('-----------------------------------------------------')
            #print('dtype grad {}'.format(param.grad.data.dtype))
            if self.weight_decay > 0:
                grad = param.grad.data + self.weight_decay * param.data
            else :
                grad = param.grad.data
            if self.u.get(param) is None:
                self.u[param] = 0
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            #print('type param.data {} u[param].data {}'.format(param.data.dtype,self.u[param].dtype))
            param.data = param.data - self.lr * Tensor(self.u[param],dtype=float32)

        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bias_correction = True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.bias_correction = bias_correction
        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if self.m.get(w) is None:
                self.m[w] = 0
            if self.v.get(w) is None:
                self.v[w] = 0

            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)
            biased_m = self.m[w] / (1 - self.beta1 ** self.t)
            biased_v = self.v[w] / (1 - self.beta2 ** self.t)
            w.data = Tensor(w.data - self.lr * biased_m / (biased_v**0.5 + self.eps),dtype=float32)
        ### END YOUR SOLUTION
