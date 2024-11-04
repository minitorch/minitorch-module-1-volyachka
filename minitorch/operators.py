"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

def mul(x: float, y: float) -> float:
    """Returns the product of x and y."""
    return x * y

def id(x: float) -> float:
    """Returns the identity of x."""
    return x

def add(x: float, y: float) -> float:
    """Returns the sum of x and y."""
    return x + y

def neg(x: float) -> float:
    """Returns the negation of x."""
    return -x

def lt(x: float, y: float) -> bool:
    """Returns True if x is less than y, otherwise False."""
    return x < y

def eq(x: float, y: float) -> bool:
    """Returns True if x is equal to y, otherwise False."""
    return x == y

def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y

def is_close(x: float, y: float) -> bool:
    """Returns True if x and y are close within a given tolerance."""
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    """Returns the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """Returns the ReLU of x."""
    return max(0, x)

def log(x: float) -> float:
    """Returns the natural logarithm of x."""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive values.")
    return math.log(x)

def exp(x: float) -> float:
    """Returns e raised to the power of x."""
    return math.exp(x)

def log_back(x: float, y: float) -> float:
    """Returns the derivative of log with respect to its input."""
    return y / x

def inv(x: float) -> float:
    """Returns the multiplicative inverse of x."""
    if x == 0:
        raise ValueError("Division by zero.")
    return 1 / x

def inv_back(x: float, y: float) -> float:
    """Returns the derivative of the inverse function."""
    return -y / (x ** 2)

def relu_back(x: float, y: float) -> float:
    """Returns the derivative of ReLU with respect to its input."""
    return y if x > 0 else 0.0

def sigmoid_back(x: float, y: float):
    """Returns the derivative of sigmoid with respect to its input."""
    return y * math.exp(x) / (1.0 + math.exp(x))

# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

from typing import List, Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

def map(func: Callable[[float], float]):
    """Applies func to each element in lst."""
    return lambda list: [func(x) for x in list]

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies func to each pair of elements from lst1 and lst2."""
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]):
        return [fn(x, y) for (x, y) in zip(ls1, ls2)]
    return _zipWith

def reduce(func: Callable[[T, T], T], initial: T):
    """Applies func cumulatively to the elements in lst, starting from initial."""
    def _reduce(lst: Iterable[T]):
        result = initial
        for x in lst:
            result = func(result, x)
        return result

    return _reduce

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates each element in lst."""
    return map(neg)(ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    """Sums all elements in lst."""
    return reduce(add, 0.0)(ls)

def prod(lst: Iterable[float]) -> float:
    """Multiplies all elements in lst."""
    return reduce(mul, 1.0)(lst)

