import cmath
import numpy as np

def select_device_simple(dev="cpu"):
    if dev == "cpu":
        return "/cpu:0"
    else:
        return "/gpu:0"

def dist(x1, x2):
    return np.linalg.norm(x1 - x2)

def mindist_1d(x1, x2):
    return dist(x1, x2), x1, x2

# def compare_exact(x1, x2,
#                   rel_tol=1e-9,
#                   abs_tol=0.0,
#                   debug=False):
#     # this needs to be rewritte, maybe use minperdist
#     if debug:
#         assert x1.size == x2.size, "x1.size: {} x2.size: {}".format(x1.size, x2.size)
#     same = isCloseArray(np.sort(np.abs(x1)), np.sort(np.abs(x2)),
#                         rel_tol=rel_tol, abs_tol=abs_tol)
#     return same

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


def database_stats(system, db, x_test, y_test, fname='dg.pdf', **kwargs):

    pot = system.get_potential()

    print "Nminima = ", len(db.minima())
    print "Nts = ", len(db.transition_states())

    make_disconnectivity_graph(system, db, x_test, y_test, fname=fname, **kwargs)

    print "Minimum Energy, RMS grad: "
    for m in db.minima():
        print m.energy, np.linalg.norm(pot.getEnergyGradient(m.coords)[1]/np.sqrt(m.coords.size))


def run_double_ended_connect(system, database, strategy='random'):
    # connect the all minima to the lowest minimum
    from pele.landscape import ConnectManager
    manager = ConnectManager(database, strategy=strategy)
    for i in xrange(database.number_of_minima()-1):
        min1, min2 = manager.get_connect_job()
        connect = system.get_double_ended_connect(min1, min2, database)
        connect.connect()


def make_disconnectivity_graph(system, database, x_test, y_test, fname='dg.pdf', **kwargs):
    import matplotlib.pyplot as plt
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, **kwargs)
    dg.calculate()

    # color DG points by test-set error
    minimum_to_testerror = lambda m: system.pot.test_model(m.coords, x_test, y_test)
    dg.color_by_value(minimum_to_testerror, colormap=plt.cm.ScalarMappable(cmap='YlGnBu').get_cmap())
    dg.plot(linewidth=1.5)
    plt.savefig(fname)