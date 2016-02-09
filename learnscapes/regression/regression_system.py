from __future__ import division
import numpy as np
import cmath
from tensorflow.examples.tutorials.mnist import input_data

from pele.systems import BaseSystem
from pele.systems.basesystem import dict_copy_update
from pele.optimize import lbfgs_cpp
from pele.landscape import smooth_path
from pele.takestep import RandomCluster, RandomDisplacement
from pele.storage import Database
from learnscapes.regression import DoubleLogisticRegressionPotential
from learnscapes.utils import isCloseArray

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


def dist(x1, x2):
    return np.linalg.norm(x1 - x2)


def mindist_1d(x1, x2):
    return dist(x1, x2), x1, x2


class RegressionSystem(BaseSystem):
    def __init__(self, x_train_data, y_train_data, hnodes, reg=0.1, dtype='float32', device='cpu'):
        BaseSystem.__init__(self)
        self.x_train_data = np.array(x_train_data, dtype=dtype)
        self.y_train_data = np.array(y_train_data, dtype=dtype)
        self.hnodes = hnodes
        self.dtype = dtype
        self.device = device
        self.reg = reg
        self.ndim = (self.y_train_data.shape[1]*self.hnodes + self.hnodes*self.x_train_data.shape[1] + self.hnodes + self.y_train_data.shape[1])
        self.pot = self.get_potential(dtype=self.dtype, device=self.device)
        self.setup_params(self.params)

    def setup_params(self, params):
        params.takestep.verbose = True
        lcp = params.double_ended_connect.local_connect_params
        lcp.pushoff_params.update(dict(stepmin=0.5))
        lcp.tsSearchParams.lowestEigenvectorQuenchParams.update(dict(dx=0.5e-3))

        nebparams = lcp.NEBparams
        # nebparams.image_density = 0.8
        # nebparams.iter_density = 50.
        # nebparams.reinterpolate = 50
        nebparams.adaptive_nimages = True
        nebparams.adaptive_niter = True
        nebparams.adjustk_freq = 10
        nebparams.k = 2000

        params.structural_quench_params.tol = 1e-5
        params.structural_quench_params.maxstep = 1
        params.structural_quench_params.M = 4
        params.structural_quench_params.iprint=100
        params.structural_quench_params.verbosity=0

        params.database.overwrite_properties = False

        # params.basinhopping.insert_rejected = True
        params.basinhopping.temperature = 10000

        lcp.tsSearchParams.hessian_diagonalization = False

        # params.database.accuracy = 1e-5

    def get_system_properties(self):
        return dict(potential="RegressionPotential",
                    x_train_data=self.x_train_data,
                    y_train_data=self.y_train_data,
                    reg=self.reg,
                    hnodes=self.hnodes,
                    )

    def get_minimizer(self, **kwargs):
        """return a function to minimize the structure

        Notes
        The function should be one of the optimizers in `pele.optimize`, or
        have similar structure.

        See Also
        --------
        pele.optimize
        """
        pot = self.get_potential()
        kwargs = dict_copy_update(self.params["structural_quench_params"], kwargs)
        return lambda coords: lbfgs_cpp(coords, pot, **kwargs)

    def get_potential(self, dtype='float32', device='cpu'):
        try:
            return self.pot
        except AttributeError:
            self.pot = DoubleLogisticRegressionPotential(self.x_train_data, self.y_train_data, self.hnodes,
                                                         reg=self.reg, dtype=dtype, device=device)
            return self.pot

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
    def get_metric_tensor(self, coords):
        return None
    
    def get_mindist(self):
        return lambda x1, x2 : mindist_1d(x1, x2)

    # def get_compare_exact(self):
    #     """
    #     are they the same minima?
    #     """
    #     return lambda x1, x2 : compare_exact(x1, x2, rel_tol=1e-5, abs_tol=1e-7, debug=False)

    def smooth_path(self, path, **kwargs):
        mindist = self.get_mindist()
        return smooth_path(path, mindist, **kwargs)

    def get_random_configuration(self):
        return np.random.normal(0, scale=1, size=self.ndim)

    def create_database(self, *args, **kwargs):
        return BaseSystem.create_database(self, *args, **kwargs)

    def get_takestep(self, **kwargs):
        """return the takestep object for use in basinhopping, etc."""
        return RandomDisplacement(stepsize=1.5)

    def draw(self, coords, index):
        pass

    def get_metric_tensor(self, coords):
        return None

    def get_nzero_modes(self):
        return 0

    def get_pgorder(self, coords):
        return 1

# def run_gui(x_train_data, y_train_data, reg=0.01):
#     from pele.gui import run_gui
#     system = RegressionSystem(x_train_data, y_train_data, reg=reg)
#     run_gui(system)

def run_gui_db(dbname="regression_logit_mnist.sqlite"):
    from pele.gui import run_gui
    try:
        db = Database(dbname, createdb=False)
        x_train_data=db.get_property("x_train_data").value(),
        y_train_data=db.get_property("y_train_data").value(),
        hnodes=db.get_property("hnodes").value(),
        reg=db.get_property("reg").value(),
    except IOError:
        pass
    hnodes, reg = hnodes[0], reg[0]
    x_train_data, y_train_data = np.array(np.array(x_train_data)[0,:,:]), np.array(np.array(y_train_data)[0,:,:])
    print np.array(x_train_data).shape, np.array(y_train_data).shape
    system = RegressionSystem(x_train_data, y_train_data, hnodes, reg=reg)
    run_gui(system, db=dbname)
    # run_double_ended_connect(system, db, strategy='random')
    # from pele.thermodynamics import get_thermodynamic_information
    # get_thermodynamic_informationormation(system, db, nproc=1, verbose=True)

def run_double_ended_connect(system, database, strategy='random'):
    # connect the all minima to the lowest minimum
    from pele.landscape import ConnectManager
    manager = ConnectManager(database, strategy=strategy)
    for i in xrange(database.number_of_minima()-1):
        min1, min2 = manager.get_connect_job()
        connect = system.get_double_ended_connect(min1, min2, database)
        connect.connect()

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = 1000
    trX, trY, teX, teY = mnist.train.images[::int(50000/bs)], mnist.train.labels[::int(50000/bs)], mnist.test.images, mnist.test.labels
    reg=0.01
    hnodes = 10
    bh = False
    if bh:
        system = RegressionSystem(trX, trY, hnodes, reg=reg, device='gpu')
        db = system.create_database("regression_logit_mnist_batch{}.sqlite".format(bs))
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(10000)
        # run_gui(system, db="regression_logit_mnist_batch{}.sqlite".format(bs))

    # if not bh:
    if True:
        run_gui_db(dbname="regression_logit_mnist_batch{}.sqlite".format(bs))

    if bh:
        # compare_minima = lambda m1, m2 : compare_exact(np.sort(np.abs(m1.coords)), np.sort(np.abs(m2.coords)), rel_tol=1e-5, debug=False)
        db = Database("regression_logit_mnist_batch{}.sqlite".format(bs))
        minima = db.minima()
        minima.sort(key=lambda m: m.energy)
        for m in minima:
           print m.energy#, m.coords
        # print minima[0].energy, np.sort(np.abs(minima[0].coords))
        # print minima[1].energy, np.sort(np.abs(minima[1].coords))
        # print compare_minima(minima[2],minima[3])


if __name__ == "__main__":
    main()