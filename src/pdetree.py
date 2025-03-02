from enum import Enum
from dataclasses import dataclass
import ast
import transforms

class ArithOp(Enum):
    PLUS = 1
    MINUS = 2
    TIMES = 3
    DIV = 4
    def __repr__(self) -> str:
        if self.value == ArithOp.PLUS.value:
            return "+"
        if self.value == ArithOp.MINUS.value:
            return "-"
        if self.value == ArithOp.TIMES.value:
            return "*"
        if self.value == ArithOp.DIV.value:
            return "/"

DIFFERENTIAL_OPERATOR_TOKEN = "d"

@dataclass
class PdetNode:
    """ Base class for PDE tree"""

@dataclass
class Expr(PdetNode):
    """ Base class for expressions """

@dataclass
class Var(Expr):
    """ Base class for variables """

@dataclass
class VarX(Var):
    """ Input variable """
    id: int
    def __repr__(self) -> str:
        return f"x{self.id}"

@dataclass
class VarU(Var):
    """ Target variable """
    def __repr__(self) -> str:
        return "u"

@dataclass
class BinOp(Expr):
    """ Base binary operations """
    left: Expr
    right: Expr
    op: ArithOp
    def __repr__(self) -> str:
        return f"{repr(self.left)} {repr(self.op)} {repr(self.right)}"

@dataclass
class Diff(Expr):
    """ Differential operator """
    num: Expr
    den: Expr
    def __repr__(self) -> str:
        return f"{DIFFERENTIAL_OPERATOR_TOKEN}({repr(self.num)}, {repr(self.den)})"

@dataclass
class Constant(Expr):
    val: float
    def __repr__(self) -> str:
        return repr(self.val)

class NodeVisitor(object):
    """ Node visitor base class similar to `ast.NodeVisitor` """
    def visit(self, node: PdetNode):
        """ Visit a node """
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: PdetNode):
        """ Visit if no explicit visitor function exists for a node """
        for _, value in node.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, PdetNode):
                        self.visit(item)
            elif isinstance(value, PdetNode):
                self.visit(item)

class Converter(ast.NodeVisitor):
    """ Convert from python AST to PDE Tree AST """
    def visit_Module(self, node: ast.Module) -> Expr:
        """ Visitor module """
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise SyntaxError("Expected exactly one expression.")
        return self.visit(node.body[0])

    def visit_Expression(self, node: ast.Expr) -> Expr:
        """ Visit expression """
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Var:
        """ Visit variable """
        if node.id.startswith("x"):
            try:
                var_id = int(node.id[1:])
            except ValueError as e:
                raise SyntaxError("Expected variable 'x' to be followed by integer.") from e
            return VarX(id=var_id)
        if node.id == "u":
            return VarU()
        raise SyntaxError(f"Variables with unknown name {node.id}")
        
    def visit_BinOp(self, node: ast.BinOp) -> BinOp:
        """ Visit binary operation """
        if isinstance(node.op, ast.Add):
            op = ArithOp.PLUS
        elif isinstance(node.op, ast.Sub):
            op = ArithOp.MINUS
        elif isinstance(node.op, ast.Mult):
            if not (isinstance(node.left, ast.Constant) or isinstance(node.right, ast.Constant)):
                raise ValueError("Not a linear PDE multiplication by expression.")
            op = ArithOp.TIMES
        elif isinstance(node.op, ast.Div):
            if not (isinstance(node.left, ast.Constant) or isinstance(node.right, ast.Constant)):
                raise ValueError("Not a linear PDE division by expression.")
            op = ArithOp.DIV
        else:
            raise SyntaxError(f"Unsupported arithmetic operation {node.op}.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return BinOp(left=left, right=right, op=op)
    
    def visit_Call(self, node: ast.Call) -> Diff:
        """ Visit differential operator """
        if not (isinstance(node.func, ast.Name) and node.func.id == DIFFERENTIAL_OPERATOR_TOKEN):
            raise SyntaxError("Expected differential operator as function.")
        if len(node.args) != 2:
            raise SyntaxError("Expected differential operator to have exactly two arguments.")
        num = self.visit(node.args[0])
        den = self.visit(node.args[1])
        if not isinstance(den, VarX):
            raise Exception("Expected differential of an input variable.")
        return Diff(num=num, den=den)

    def visit_Constant(self, node: ast.Constant) -> Constant:
        """ Visit constant"""
        return Constant(val=node.value)

    def generic_visit(self, node):
        raise SyntaxError(f"Unsupported node {node}")


def parse(pde_expression: str) -> PdetNode:
    """ Parse differential equation to PDE Tree """
    tree = ast.parse(pde_expression, mode='eval')
    tree = transforms.ConstantFold().visit(tree)
    pde_tree = Converter().visit(tree)
    return pde_tree
