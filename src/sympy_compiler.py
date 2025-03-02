import sympy
import pdetree
import linalg as spla

class LinOpGenerator(pdetree.NodeVisitor):
    """
    Generates a Linear Operator function that returns a composed
    function of two arguments kernel and symbol
    """
    def visit_Diff(self, node: pdetree.Diff):
        """ Creates a diff function that takes kernel and variable """
        num_fn = self.visit(node.num)
        den_fn = self.visit(node.den)
        return lambda k, x: sympy.diff(num_fn(k, x), den_fn(k, x))

    def visit_VarU(self, _: pdetree.VarU):
        """ Returns the kernel function """
        return lambda k, _: k

    def visit_VarX(self, node: pdetree.VarX):
        """ Returns the variable corresponding to its id """
        return lambda _, x: x[node.id]

    def visit_BinOp(self, node: pdetree.BinOp):
        """ Returns the binary operation """
        left_fn = self.visit(node.left)
        right_fn = self.visit(node.right)
        if node.op.value == pdetree.ArithOp.PLUS.value:
            return lambda k, x: left_fn(k, x) + right_fn(k,x)
        if node.op.value == pdetree.ArithOp.MINUS.value:
            return lambda k, x: left_fn(k, x) - right_fn(k,x)
        if node.op.value == pdetree.ArithOp.TIMES.value:
            return lambda k, x: left_fn(k, x) * right_fn(k,x)
        if node.op.value == pdetree.ArithOp.DIV.value:
            return lambda k, x: left_fn(k, x) / right_fn(k,x)

    def visit_Constant(self, node: pdetree.Constant):
        """ Returns a constant value """
        return lambda _, __: node.val

def pdet_compile(node: pdetree.PdetNode, kernel: sympy.Symbol, sym_x: spla.SymVector, sym_y: spla.SymVector):
    """ 
    Compiles the the differential equation to 
    corresponding sympy equation. 
    """
    linop = LinOpGenerator().visit(node)
    lx = sympy.simplify(linop(kernel, sym_x))
    ly = sympy.simplify(linop(kernel, sym_y))
    lxy = sympy.simplify(linop(lx, sym_y))
    return (lxy, lx, ly)
