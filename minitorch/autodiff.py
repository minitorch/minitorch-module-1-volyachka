from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_temp1, vals_temp2 = list(vals), list(vals)
    vals_temp1[arg] += epsilon
    f_1 = f(*vals_temp1)
    vals_temp2[arg] -= epsilon
    f_2 = f(*vals_temp2)
    return (f_1 - f_2) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.

    visited = set()
    output = []

    def dfs(variable) -> None:

        if variable.is_constant():
            return
        
        if variable.unique_id in visited:
            return
        
        visited.add(variable.unique_id)

        if not variable.is_leaf():
            for parent in variable.parents:
                if parent.unique_id not in visited:
                    dfs(parent)
        output.append(variable)
        return

    dfs(variable)
    return reversed(output)



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    derivatives = dict()
    derivatives[variable.unique_id] = deriv

    top_sort = topological_sort(variable)
    for var in top_sort:
        if var.history.last_fn is None:
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            for parent, derivate in var.chain_rule(derivatives[var.unique_id]):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0
                derivatives[parent.unique_id] += derivate


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values