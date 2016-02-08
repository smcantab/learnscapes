from __future__ import division
import numpy as np
import cmath
from tensorflow.examples.tutorials.mnist import input_data

from pele.systems import BaseSystem
from pele.systems.basesystem import dict_copy_update
from pele.optimize import lbfgs_cpp
from pele.landscape import smooth_path
from pele.takestep.generic import TakestepSlice
from pele.storage import Database
from learnscapes.regression import RegressionPotential, LogisticRegressionGraph
from learnscapes.utils import isClose

def compare_exact(x1, x2,
                  rel_tol=1e-9,
                  abs_tol=0.0,
                  method='weak',
                  debug=False):
    # this needs to be rewritte, maybe use minperdist
    # N = x1.size
    # if debug:
    #     assert x1.size == x2.size, "x1.size: {} x2.size: {}".format(x1.size, x2.size)
    # dot = np.dot(x1, x2)
    # same =(isClose(dot, N, rel_tol=rel_tol, abs_tol=abs_tol, method=method) or
    #        isClose(dot, -N, rel_tol=rel_tol, abs_tol=abs_tol, method=method))
    # return same

def dist(x1, x2):
    return np.linalg.norm(x1 - x2)


def mindist_1d(x1, x2):
    d1 = dist(x1, x2)
    d2 = dist(x1, -x2)
    if d1 < d2:
        return d1, x1, x2
    else:
        return d2, x1, -x2

class UniformPSpinSPhericalRandomDisplacement(TakestepSlice):

    def __init__(self, ndim, stepsize=0.5):
        TakestepSlice.__init__(self, stepsize=stepsize)
        self.ndim = ndim

    def takeStep(self, coords, **kwargs):
        assert len(coords) == self.ndim
        # coords[self.srange] += np.random.uniform(low=-self.stepsize, high=self.stepsize, size=coords[self.srange].shape)
        coords[self.srange] = np.random.normal(0, scale=0.01, size=coords[self.srange].shape)*self.stepsize

class RegressionSystem(BaseSystem):
    def __init__(self, x_train_data, y_train_data, graph_type, reg=0.01, dtype='float32', device='cpu'):
        BaseSystem.__init__(self)
        self.x_train_data = np.array(x_train_data)
        self.y_train_data = np.array(y_train_data)
        self.graph_type = graph_type
        self.dtype = dtype
        self.device = device
        self.reg = reg
        self.ndim = self.y_train_data.shape[1]*(self.x_train_data.shape[1]+1)
        self.pot = self.get_potential(dtype=self.dtype, device=self.device)
        self.setup_params(self.params)

    def setup_params(self, params):
        params.takestep.verbose = True
        nebparams = params.double_ended_connect.local_connect_params.NEBparams
        nebparams.image_density = 0.8
        nebparams.iter_density = 50.
        nebparams.reinterpolate = 50
        nebparams.adaptive_nimages = True
        nebparams.adaptive_niter = True
        nebparams.adjustk_freq = 10
        nebparams.k = 2000
        params.structural_quench_params.tol = 1e-7
        params.structural_quench_params.maxstep = 1
        params.structural_quench_params.M = 4
        params.structural_quench_params.iprint=10
        params.structural_quench_params.verbosity=0

        params.database.overwrite_properties = False
        
        params.basinhopping.insert_rejected = True
        params.basinhopping.temperature = 10000
        
        tsparams = params.double_ended_connect.local_connect_params.tsSearchParams
        tsparams.hessian_diagonalization = False

    def get_system_properties(self):
        return dict(potential="RegressionPotential",
                    x_train_data=self.x_train_data,
                    y_train_data=self.y_train_data,
                    reg=self.reg,
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
            self.pot = RegressionPotential(self.x_train_data, self.y_train_data,
                                           self.graph_type, reg=self.reg, dtype=dtype,
                                           device=device)
            return self.pot

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
    def get_metric_tensor(self, coords):
        return None
    
    # def get_nzero_modes(self):
    #     return 0
    #
    # def get_pgorder(self, coords):
    #     return 1
    
    def get_mindist(self):
        return lambda x1, x2 : mindist_1d(x1, x2)

    def get_compare_exact(self):
        """
        are they the same minima?
        """
        return lambda x1, x2 : compare_exact(x1, x2, rel_tol=1e-6, abs_tol=0.0,
                                             method='weak', debug=True)

    def smooth_path(self, path, **kwargs):
        mindist = self.get_mindist()
        return smooth_path(path, mindist, **kwargs)

    def get_random_configuration(self):
        return np.random.normal(0, scale=0.01, size=self.ndim)

    def create_database(self, *args, **kwargs):
        return BaseSystem.create_database(self, *args, **kwargs)

    def get_takestep(self, **kwargs):
        """return the takestep object for use in basinhopping, etc.
        
        default is random displacement with adaptive step size 
        adaptive temperature
        
        See Also
        --------
        pele.takestep
        """
        kwargs = dict(self.params["takestep"].items() + kwargs.items())
        try:
            stepsize = kwargs.pop("stepsize")
        except KeyError:
            stepsize = 1 # this is a completely random value
        return UniformPSpinSPhericalRandomDisplacement(self.ndim, stepsize=stepsize)

    def draw(self, coords, index):
        pass


# def run_gui(x_train_data, y_train_data, graph_type, reg=0.01):
#     from pele.gui import run_gui
#     graph_type = LogisticRegressionGraph
#     system = RegressionSystem(x_train_data, y_train_data, graph_type, reg=reg)
#     run_gui(system)


def run_gui_db(dbname="regression_logit_mnist.sqlite"):
    from pele.gui import run_gui
    try:
        db = Database(dbname, createdb=False)
        x_train_data=db.get_property("x_train_data").value()[0],
        y_train_data=db.get_property("y_train_data").value()[0],
        reg=db.get_property("reg").value(),
    except IOError:
        pass
    graph_type = LogisticRegressionGraph
    print np.array(x_train_data).shape, np.array(y_train_data).shape
    system = RegressionSystem(x_train_data, y_train_data, graph_type, reg=reg)
    run_gui(system, db=dbname)


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = 1000
    trX, trY, teX, teY = mnist.train.images[:bs], mnist.train.labels[:bs], mnist.test.images, mnist.test.labels
    reg=0.01
    from pele.gui import run_gui
    if True:
        graph_type = LogisticRegressionGraph
        system = RegressionSystem(trX, trY, graph_type, reg=reg, device='gpu')
        db = system.create_database("regression_logit_mnist_batch{}.sqlite".format(bs))
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(20)
        # run_gui(system, db="regression_logit_mnist_batch{}.sqlite".format(bs))

    if False:
        run_gui_db(dbname="regression_logit_mnist_batch{}.sqlite".format(bs))

    if True:
        compare_minima = lambda m1, m2 : compare_exact(np.sort(np.abs(m1.coords)), np.sort(np.abs(m2.coords)), rel_tol=1e-6, debug=False)
        db = Database("regression_logit_mnist_batch{}.sqlite".format(bs))
        minima = db.minima()
        minima.sort(key=lambda m: m.energy)
        # for m in minima:
        #    print m.energy, m.coords
        print minima[0].energy, np.sort(np.abs(minima[0].coords))
        print minima[1].energy, np.sort(np.abs(minima[1].coords))
        print compare_minima(minima[0],minima[1])

