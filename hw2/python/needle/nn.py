"""The module.
"""
import random
from typing import List, Callable, Any
from python.needle.autograd import Tensor
from python.needle import ops, relu
import python.needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # print("-----------------------------------------------------------")
        # print('X.shape {} bias.shape {} weight.shape {}'.format(X.shape,self.bias.shape,self.weight.shape))
        # X (batch_size,in_features) weight (in_feature,out_features) bias (out_features,1)
        ### BEGIN YOUR SOLUTION
        out = X.matmul(self.weight)
        if self.bias: # check whether the bias is None
            out = out + self.bias.broadcast_to(out.shape) # broadcast_to [batch_size,out_features]
        return out
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0],-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for sub_module in self.modules:
            x = sub_module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # batch_size logit.shape[0] class_number logits.shape[-1]
        y_encode_one_hot = init.one_hot(logits.shape[-1],y)
        # 先计算防止溢出的logsum维度为(batch_size,1) 再计算true_label的值通过logits * y_encode_one_hot求得
        return (ops.logsumexp(logits,axes=(-1,)) - (logits * y_encode_one_hot).sum(axes=(1,)).reshape((-1,1))).sum() / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim),requires_grad=True)
        self.bias = Parameter(init.zeros(dim),requires_grad=True)
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:

            # print('----------------------------------------------')
            # print('running_mean.shape {}'.format(self.running_mean.shape))
            norm = (x - self.running_mean.broadcast_to(x.shape))/((self.running_var.broadcast_to(x.shape) +self.eps) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)

        else :
            mean = x.sum((0,)) / x.shape[0]
            std = ((x - mean.broadcast_to(x.shape)) ** 2).sum((0,)) / x.shape[0]
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * std.data
            norm = (x - mean.broadcast_to(x.shape)) / ((std.broadcast_to(x.shape) + self.eps ) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(dim, requires_grad=True))
        self.b = Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x [batch_size,dim]
        mean = (x.sum((1,)) / x.shape[1]).reshape((-1,1)).broadcast_to(x.shape)

        std  = (((x - mean) ** 2).sum((1,))/x.shape[1]).reshape((-1,1)).broadcast_to(x.shape)
        #std = x.realize_cached_data().std(axis=1,keepdims=True)
        # 注意这里面没有自动广播，得自己手动设置
        return   (x - mean) / ((std + self.eps) ** 0.5) * self.w.broadcast_to(x.shape)  + self.b.broadcast_to(x.shape)
        #raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # 随机失活原来对应的是输出的值随机一部分变为0
            mask = init.randb(*x.shape, p=1-self.p)
            return x * mask / (1 - self.p)
        else :
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



