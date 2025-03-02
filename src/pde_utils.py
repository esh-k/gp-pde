import torch
import kernels

def sample_uniform_grid(dim_ranges: torch.Tensor, n_domain, boundary_spec):
    """ 
    Sample uniformly from a n-dimensional grid and its boundary
    
    `dim_ranges`: tensor with two rows, 
        the first row is min and second row is the max 
        with the column index corresponding to the the dimension
    
    `n_domain`: number of samples in the domain (within the are of the grid)
    
    `boundary_spec`: list of (number of samples along condition, dimension_condition)
        where, dimension_condition: is a tuple of floats corresponding to fixed dimension,
                    None corresponds to free dimension from which samples will be drawn.
    """
    n_dims = dim_ranges.shape[-1]
    a, b = dim_ranges
    slopes = b - a
    x_domain = slopes * torch.rand((n_domain, n_dims)) + a
    x_boundary_full = []
    for spec in boundary_spec:
        n_samples, dim_spec = spec
        if n_dims != len(dim_spec):
            raise ValueError("Dimension mismatch between `dim_ranges` and `boundary_spec`")
        x_boundary_ls = []
        for i in range(n_dims):
            if dim_spec[i] is not None:
                x_boundary_ls.append(torch.full((n_samples, 1), dim_spec[i]))
            else:
                samples = slopes[i] * torch.rand((n_samples,1)) + a[i]
                x_boundary_ls.append(samples)
        x_boundary_full.append(torch.hstack(x_boundary_ls))
    x_boundary = torch.vstack(x_boundary_full)
    return x_domain, x_boundary

def construct_joint_kernel_from_linear_ops(lxy, lx, kernel, x_domain, x_boundary, condition=1e-4):
    """ Construct a joint kernel from linear operations
    `lxy`:  Linear operation on both x and y argument
    `lx`:   Linear operation on only x argument
    `kernel`: Kernel function
    `x_domain`: the domain of the solution
    `x_boundary`: the boundary of the solution
    `condition`: Conditioning is often poor on joint kernels matrices.
        To improve, set the condition value accordingly.
        To remove conditioning, set to 0.
    """
    top_left_in = x_domain
    bottom_right_in = x_boundary
    top_left = lxy(top_left_in, top_left_in)
    top_right = lx(top_left_in, bottom_right_in)
    bottom_left = top_right.T
    bottom_right = kernel(bottom_right_in, bottom_right_in)
    trace1 = torch.trace(top_left)
    trace2 = torch.trace(bottom_right)
    ratio = trace1 / trace2
    theta = torch.vstack([torch.hstack([top_left, top_right]),torch.hstack([bottom_left, bottom_right])])
    diag = torch.diag(condition * (torch.hstack([
        ratio * torch.ones(top_left.shape[0]),
        torch.ones(bottom_right.shape[0])
    ])))
    theta = theta + diag
    return theta

def gen_grid_points(dim_ranges: torch.Tensor, n_points):
    """ Generate a uniform grid of points 
    `dim_ranges`: the range of along each dimension
    `n_points`: Number of points in the grid
    """
    n_dims = dim_ranges.shape[-1]
    a, b = dim_ranges
    spaces = [torch.linspace(a[i], b[i], steps=n_points) for i in range(n_dims)]
    return torch.meshgrid(spaces, indexing="xy")

def infer_from_kernel(theta: torch.Tensor, ly: kernels.SympyKernel, kernel: kernels.SympyKernel, x_domain: torch.Tensor, x_boundary: torch.Tensor, u_train: torch.Tensor, x_test: torch.Tensor):
    """Performs GP inference on a given training kernel matrix

    Args:
        theta (torch.Tensor): joint training kernel matrix 
        ly (kernels.SympyKernel): linear operation on y argument of kernel
        kernel (kernels.SympyKernel): kernel module
        x_domain (torch.Tensor): training domain
        x_boundary (torch.Tensor): training boundary
        u_train (torch.Tensor): output for the combined [x_domain, x_boundary]
        x_test (torch.Tensor): the test input

    Returns:
        torch.Tensor: Inferred mean
    """
    L = torch.linalg.cholesky(theta)
    k_test_train = torch.hstack([ly(x_test, x_domain), kernel(x_test, x_boundary)])
    test_mean = k_test_train @ torch.cholesky_solve(u_train, L)
    return test_mean