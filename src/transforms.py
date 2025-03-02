import ast

class ConstantFold(ast.NodeTransformer):
    """ Convert from python AST to PDE Tree AST """
    # TODO: Allow rearranging operators with same precedence
    def visit_BinOp(self, node: ast.BinOp) -> ast.Expr:
        """ Visit binary operation """
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            result = eval(compile(ast.Expression(node), '<string>', 'eval')) # type: ignore
            return ast.Constant(result)
        return node
