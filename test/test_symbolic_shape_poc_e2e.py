from torch._C import _disabled_torch_function_impl
from torch.testing._internal.common_utils import run_tests, TestCase
import unittest
import torch
from torch.utils._pytree import tree_map
import torch._decomp
from torch._meta_registrations import register_meta, meta_funcs
aten = torch.ops.aten

try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
skipIfNoSympy = unittest.skipIf(not HAS_SYMPY, "no sympy")


@register_meta([aten.add.Tensor, aten.sub.Tensor])
def binary_meta(a, b):
    return a.new_empty(a.size())


@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    concat_length = 0
    shape = tensors[0].shape
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length
    new_shape = list(shape)
    new_shape[dim] = concat_length
    return tensors[0].new_empty(new_shape)


@register_meta([aten.narrow_copy.SymInt])
def narrow_copy_symint_meta(a, dim, start, length, **kwargs):
    shape = []
    for i, x in enumerate(a.size()):
        if i == dim:
            shape.append(length)
        else:
            shape.append(x)
    return a.new_empty(tuple(shape))


@register_meta([aten.expand.SymInt])
def expand_symint_meta(a, size, implicit=False):
    return a.new_empty(size)


from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

def f(x):
    val = torch.nn.functional.gelu(x)
    return torch.cat([val, x])

fx_g = make_fx(f)(torch.randn(5, 5))
print(fx_g)

exit(0)

foo = torch.empty(shape_env.create_symint("foo", 3), device='meta')
fake_tensor_mode = FakeTensorMode()
test = FakeTensor(fake_tensor_mode, foo, 'cuda')
with fake_tensor_mode:
    print(torch.ops.aten.expand.SymInt(test, [test.shape[0], test.shape[0]]))
    # print(torch.empty(test.shape, device='meta'))
    # print(torch.cat([test, test]).shape)
print(shape_env.guards)