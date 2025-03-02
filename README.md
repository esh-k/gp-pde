# gp-pde
Frame work for solving linear Partial Differential Equations (PDE) using Gaussian Process (GP)

## Usage
Output variable: $u$ <br/>
Input variables: $x_1,...,x_n$


Refer to example notebooks in `notebooks/`

### Describing the PDE
E.g., Heat equation with non-linear boundary:
$$
\begin{align}
\Delta u(x) &= 0 & \text{$x \in \mathcal{X}$ }\\
u(x) &= g(x) & \text{$x \in \partial \mathcal{X}$ }
\end{align}
$$
where, $\mathcal{X}, \partial \mathcal{X}$ are the domain and boundary of the defined PDE.
In 2d equation $(1)$ would be described as:
$$
\frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2} = 0
$$
Which in our framework would be written as:
```
d(d(u,x0),x0) + d(d(u,x1),x1) = 0
```
The boundary condition and the domain are passed as numpy arrays or a function that acts on boundary values. The x variables start from 0 to d where d is the number of dimensions. 



### Describe the kernel function
E.g., Radial Basis Function (RBF) kernel
$$
\begin{align*}
k_{\mathrm{RBF}}(x, y) = \sigma^2 \exp\left(\frac{(x - y)^2}{2 \ell^2 }\right)
\end{align*}
$$
where, $\sigma$ is the amplitude and $\ell$ is the length scale parameters.
The kernel function is described using a function.

The framework uses [sympy](https://www.sympy.org/en/index.html) and [sympytorch](https://github.com/patrick-kidger/sympytorch) and computed using [pytorch](https://pytorch.org/).


## Design
The PDE expression is parsed to a python AST which only supports linear operations and the differential operator described by a function $d(\cdot, \cdot)$. This AST is then converted to a sympy expression by chaining functions lazily. The final output of compilation would be a function that accepts a kernel function described by a sympy expression. 
