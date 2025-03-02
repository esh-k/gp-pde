import sympy

class SymVector:
    """ Symbolic vector wrapper with better support than sympy """
    def __init__(self, base_sym: str, n_dims: int):
        """ 
        base_sym: unique vector name 
        n_dims: dimension of the vector
        """
        self.shape = (n_dims,)
        self.vec: list[sympy.Symbol] = [sympy.symbols(f'{base_sym}{i}') for i in range(n_dims)]

    def __getitem__(self, key):
        """ Get the vector element """
        if isinstance(key, slice):
            raise NotImplementedError("Slice operation not supported")
        return self.vec[key]

    def test_vector_same_size(self, b):
        """ 
        Tests if self and the other vector are the same size 
        otherwise raises an error
        """
        if len(self.vec) != len(b.vec):
            raise ValueError("Vectors should have same length to compute dot product")

    def dot(self, b):
        """ Dot product between vectors """
        self.test_vector_same_size(b)
        sum = 0.0
        for x1, x2 in zip(self.vec, b.vec):
            sum = sum + x1 * x2
        return sum

    @staticmethod
    def create_from_list(ls):
        """ Creates a new vector from list """
        res_vec = SymVector("dummy", 1)
        res_vec.shape = (len(ls),)
        res_vec.vec = ls
        return res_vec

    def handle_scalar(self, b, binop):
        """ Handle scalar operand for binary operation """
        res = []
        for x in self.vec:
            res.append(binop(x, b))
        return res

    def handle_vector(self, b, binop):
        """ Handle vector operand for binary operation """
        res = []
        for x1, x2 in zip(self.vec, b.vec):
            res.append(binop(x1, x2))
        return SymVector.create_from_list(res)

    def handle_binop(self, b, binop):
        """ Handle binary operation """
        if isinstance(b, float):
            return self.handle_scalar(b, binop)
        self.test_vector_same_size(b)
        return self.handle_vector(b, binop)

    def __add__(self, b):
        return self.handle_binop(b, lambda x, y: x + y)

    def __sub__(self, b):
        return self.handle_binop(b, lambda x, y: x - y)

    def __mul__(self, b):
        return self.handle_binop(b, lambda x, y: x * y)

    def __div__(self, b):
        return self.handle_binop(b, lambda x, y: x / y)

    def sum(self):
        """ Return sum of all elements in vector """
        s = 0.0
        for x in self.vec:
            s += x
        return s

    def __len__(self):
        return len(self.vec)
