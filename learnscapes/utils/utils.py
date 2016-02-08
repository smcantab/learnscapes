import cmath
import numpy as np

def select_device_simple(dev="cpu"):
    if dev == "cpu":
        return "/cpu:0"
    else:
        return "/gpu:0"

def isClose(a, b, rel_tol=1e-9, abs_tol=0.0, method='weak'):
    """
    code imported from math.isclose python 3.5
    """
    if method not in ("asymmetric", "strong", "weak", "average"):
        raise ValueError('method must be one of: "asymmetric",'
                         ' "strong", "weak", "average"')

    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError('error tolerances must be non-negative')

    if a == b:  # short-circuit exact equality
        return True
    # use cmath so it will work with complex or float
    if cmath.isinf(a) or cmath.isinf(b):
        # This includes the case of two infinities of opposite sign, or
        # one infinity and one finite number. Two infinities of opposite sign
        # would otherwise have an infinite relative tolerance.
        return False
    diff = abs(b - a)
    if method == "asymmetric":
        return (diff <= abs(rel_tol * b)) or (diff <= abs_tol)
    elif method == "strong":
        return (((diff <= abs(rel_tol * b)) and
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "weak":
        return (((diff <= abs(rel_tol * b)) or
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "average":
        return ((diff <= abs(rel_tol * (a + b) / 2) or
                (diff <= abs_tol)))
    else:
        raise ValueError('method must be one of:'
                         ' "asymmetric", "strong", "weak", "average"')


def isCloseArray(a, b, rel_tol=1e-9, abs_tol=0.0):
    a, b = np.array(a), np.array(b)
    assert a.size == b.size, "learnscapes isCloseArray, arrays size mismatch"
    return np.allclose(a, b, rtol=rel_tol, atol=abs_tol, equal_nan=False)
