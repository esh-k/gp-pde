"""
Different ways to create RBF kernels
"""
import sympy
import linalg as spla
import torch
from torch import nn

def rbf_kernel_sympy(
        n_dims=1,
        left_base_sym: str = "x",
        right_base_sym: str = "y",
        amplitude: float | sympy.Symbol=1.0,
        length_scale: float | sympy.Symbol=1.0) -> tuple[sympy.Symbol, spla.SymVector, spla.SymVector]:
    """
    Creates the RBF kernel in sympy 
    returns the kernel and associated symbol operands
    """
    x = spla.SymVector(left_base_sym, n_dims)
    y = spla.SymVector(right_base_sym, n_dims)
    delta = x - y
    k = amplitude**2 * sympy.exp(-delta.dot(delta)/2.0/length_scale**2)
    return k, x, y

class SympyKernel(nn.Module):
    def __init__(self, sub_module, x_var: spla.SymVector, y_var: spla.SymVector):
        super(SympyKernel, self).__init__()
        self.sub_module = sub_module
        self.x_var = x_var
        self.y_var = y_var
        if len(self.x_var) != len(self.y_var):
            raise ValueError("Expected both sym vectors to match dimensions")
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # assumes batch shape is dim 0
        # assumes vector dimension is -1
        if not (x.shape[-1] == y.shape[-1] == len(self.x_var)):
            raise ValueError("Expected dimension of input to match sym vectors")
        # create a grid of the two vectors
        n_row = x.shape[0]
        n_col = y.shape[0]
        idx1, idx2 = torch.cartesian_prod(torch.arange(n_row), torch.arange(n_col)).T
        x_ = x[idx1]
        y_ = y[idx2]
        arg_dict = { repr(self.x_var[i]):x_[:,i] for i in range(len(self.x_var)) }
        arg_dict = arg_dict | { repr(self.y_var[i]):y_[:,i] for i in range(len(self.y_var)) }
        return self.sub_module(**arg_dict).view(n_row, n_col)
