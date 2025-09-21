from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
import minitorch
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    is_close,
    exp,
    log,
    negList,
    addLists,
    sum,
    prod
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that the main operators all return the same value of the python version"""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    """Check that a - 1.0 is always less than a"""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Property-based tests for minitorch.operators.sigmoid.

    Checks:
        * Boundedness in (0, 1).
        * Complement identity: 1 - sigmoid(x) == sigmoid(-x).
        * Midpoint: sigmoid(0) == 0.5.
        * Monotonicity: strictly increasing in a neighborhood.

    Args:
        a: Random float drawn from small_floats.

    Returns:
        None
    """
    s = sigmoid(a)
    assert 0.0 <= s <= 1.0

    assert is_close(1.0 - s, sigmoid(-a), 1e-12)

    assert is_close(sigmoid(0.0), 0.5, 1e-12)

    eps = 1e-6
    if a + eps > a:
        s1 = sigmoid(a)
        s2 = sigmoid(a + eps)
        assert s2 >= s1


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Transitivity of the strict less-than relation.

    Verifies that if a < b and b < c then a < c.

    Args:
        a: First random float.
        b: Second random float.
        c: Third random float.

    Returns:
        None
    """
    if lt(a, b) and lt(b, c):
        assert lt(a, c)


@pytest.mark.task0_2
def test_symmetric() -> None:
    """Commutativity of multiplication.

    Ensures mul(a, b) == mul(b, a) for representative numeric pairs,
    including zeros, positives, negatives, and disparate magnitudes.

    Returns:
        None
    """
    samples = [
        (0.0, 0.0),
        (1.0, 2.5),
        (-3.0, 4.0),
        (-2.0, -5.0),
        (1e-6, 1e6),
    ]
    for a, b in samples:
        left = mul(a, b)
        right = mul(b, a)
        assert is_close(left, right, 1e-12)


@pytest.mark.task0_2
def test_distribute() -> None:
    """Left-distributivity of multiplication over addition.

    Confirms the identity z * (x + y) == z * x + z * y on several
    representative numeric cases.

    Returns:
        None
    """
    cases = [
        (2.0, 3.0, 4.0),
        (-1.5, 2.0, -3.0),
        (0.0, 5.0, -7.0),
        (1e-3, 1e6, -1e6),
    ]
    for z, x, y in cases:
        left = mul(z, add(x, y))
        right = add(mul(z, x), mul(z, y))
        assert is_close(left, right, 1e-12)


@pytest.mark.task0_2
def test_other() -> None:
    """Additional sanity properties for selected operators.

    Validates:
        * ReLU outputs are non-negative and equal to the input for positives,
          else zero.
        * For positive inputs a, exp(log(a)) == a (within tolerance).

    Returns:
        None
    """
    for x in [-3.0, -1e-9, 0.0, 1e-9, 5.0]:
        r = relu(x)
        assert r >= 0.0
        if x > 0:
            assert r == x
        else:
            assert r == 0

    positives = [0.1, 1.0, 3.14, 10.0]
    for a in positives:
        assert is_close(exp(log(a)), a,
                        1e-12)
        assert is_close(exp(log(a)), a)


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    sum(ls1) + sum(ls2) == sum(ls1[i] + ls2[i])
    """
    lhs = minitorch.operators.sum(ls1) + minitorch.operators.sum(ls2)
    rhs = minitorch.operators.sum(addLists(ls1, ls2))
    assert_close(lhs, rhs)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), minitorch.operators.sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
