from __future__ import division
import numpy as np
import cmath
import tensorflow as tf

from learnscapes.pspinspherical import MeanFieldPSpinSphericalTF
from pele.systems import BaseSystem
from pele.systems.basesystem import dict_copy_update
from pele.optimize import lbfgs_cpp
from pele.landscape import smooth_path
from pele.transition_states._zeroev import orthogonalize
from pele.takestep.generic import TakestepSlice
from pele.storage import Database
from learnscapes.utils import isClose

def compare_exact(x1, x2,
                  rel_tol=1e-9,
                  abs_tol=0.0,
                  method='weak',
                  even=False,
                  debug=False):
    N = x1.size
    if debug:
        assert x1.size == x2.size, "x1.size: {} x2.size: {}".format(x1.size, x2.size)
        assert isClose(np.dot(x1,x1), N), "|x1|^2 = {} != {}".format(np.dot(x1,x1), N)
        assert isClose(np.dot(x2,x2), N), "|x2|^2 = {} != {}".format(np.dot(x2,x2), N)
    dot = np.dot(x1, x2)
    if even:
        same =(isClose(dot, N, rel_tol=rel_tol, abs_tol=abs_tol, method=method) or
               isClose(dot, -N, rel_tol=rel_tol, abs_tol=abs_tol, method=method))
    else:
        same = isClose(dot, N, rel_tol=rel_tol, abs_tol=abs_tol, method=method)
    return same


def normalize_spins(x):
    x /= (np.linalg.norm(x)/np.sqrt(x.size))
    return x


def dist(x1, x2):
    return np.linalg.norm(x1 - x2)


def mindist_even(x1, x2):
    d1 = dist(x1, x2)
    d2 = dist(x1, -x2)
    if d1 < d2:
        return d1, x1, x2
    else:
        return d2, x1, -x2

def mindist_odd(x1, x2):
    return dist(x1, x2), x1, x2

def spin_mindist_1d(x1, x2, even=False):
    x1 = normalize_spins(x1)
    x2 = normalize_spins(x2)
    if even:
        return mindist_even(x1, x2)
    else:
        return mindist_odd(x1, x2)


class UniformPSpinSPhericalRandomDisplacement(TakestepSlice):

    def __init__(self, nspins, stepsize=0.5):
        TakestepSlice.__init__(self, stepsize=stepsize)
        self.nspins = nspins

    def takeStep(self, coords, **kwargs):
        assert len(coords) == self.nspins
        coords[self.srange] += np.random.uniform(low=-self.stepsize, high=self.stepsize, size=coords[self.srange].shape)
        coords[self.srange] /= np.linalg.norm(coords[self.srange])/np.sqrt(self.nspins)


class MeanFieldPSpinSphericalSystem(BaseSystem):
    def __init__(self, nspins, p=3, interactions=None, dtype='float64', device='cpu'):
        BaseSystem.__init__(self)
        self.nspins = nspins
        self.p = p
        if interactions is not None:
            self.interactions = np.array(interactions)
        else:
            self.interactions = self.get_interactions(self.nspins, self.p, dtype)
        self.pot = self.get_potential(dtype=dtype, device=device)
        self.setup_params(self.params)

    def setup_params(self, params):
        params.takestep.verbose = True
        nebparams = params.double_ended_connect.local_connect_params.NEBparams
        # nebparams.image_density = 0.8
        # nebparams.iter_density = 50.
        nebparams.reinterpolate = 50
        nebparams.adaptive_nimages = True
        nebparams.adaptive_niter = True
        nebparams.adjustk_freq = 10
        nebparams.k = 2000
        params.structural_quench_params.tol = 1e-6
        params.structural_quench_params.maxstep = self.nspins
        params.structural_quench_params.M = 4
        params.structural_quench_params.iprint=100
        params.structural_quench_params.verbosity=0

        params.database.overwrite_properties = False
        
        params.basinhopping.insert_rejected = True
        params.basinhopping.temperature = 10000
        
        tsparams = params.double_ended_connect.local_connect_params.tsSearchParams

        tsparams.hessian_diagonalization = False

    def get_system_properties(self):
        return dict(potential="PSpinSPherical_model",
                    nspins=self.nspins,
                    p=self.p,
                    interactions=self.interactions,
                    )

    def get_interactions(self, nspins, p, dtype):
        norm = tf.random_normal([nspins for _ in xrange(p)], mean=0, stddev=1.0, dtype=dtype)
        interactions = norm.eval(session=tf.Session())
        return interactions

    def get_minimizer(self, **kwargs):
        """return a function to minimize the structure

        Notes
        The function should be one of the optimizers in `pele.optimize`, or
        have similar structure.

        See Also
        --------
        pele.optimize
        """
        def event_normalize_spins(**kwargs):
            kwargs["coords"] = normalize_spins(kwargs["coords"])
        pot = self.get_potential()
        kwargs = dict_copy_update(self.params["structural_quench_params"], kwargs)
        kwargs = dict_copy_update(dict(events=[event_normalize_spins]), kwargs)
        return lambda coords: lbfgs_cpp(coords, pot, **kwargs)

    def get_potential(self, dtype='float64', device='cpu'):
        try:
            return self.pot
        except AttributeError:
            self.pot = MeanFieldPSpinSphericalTF(self.interactions, self.nspins, self.p, dtype=dtype, device=device)
            return self.pot

    def _orthog_to_zero(self, v, coords):
        zerov = [np.array(coords)/np.linalg.norm(coords)]
        return orthogonalize(v, zerov)
#
    def get_orthogonalize_to_zero_eigenvectors(self):
        return self._orthog_to_zero
    
    def get_metric_tensor(self, coords):
        return None
    
    def get_nzero_modes(self):
        return 1

    def get_pgorder(self, coords):
        return 1
    
    def get_mindist(self):
        even = self.p % 2 == 0
        return lambda x1, x2 : spin_mindist_1d(x1, x2, even=even)

    def get_compare_exact(self):
        """
        are they the same minima?
        """
        even = self.p % 2 == 0
        return lambda x1, x2 : compare_exact(x1, x2, rel_tol=1e-4, abs_tol=0.0,
                                             method='weak', even=even, debug=True)

    def smooth_path(self, path, **kwargs):
        mindist = self.get_mindist()
        return smooth_path(path, mindist, **kwargs)

    def get_random_configuration(self):
        coords = np.random.normal(0, 1, self.nspins)
        return normalize_spins(coords)

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
            stepsize = np.sqrt(self.nspins)/2
        return UniformPSpinSPhericalRandomDisplacement(self.nspins, stepsize=stepsize)

    def draw(self, coords, index):
        pass


def normalize_spins_db(db):
    for m in db.minima():
        x = normalize_spins(m.coords)
        print np.max(x), np.min(x)
        m.coords = x
    db.session.commit()


def run_gui(N, p):
    from pele.gui import run_gui
    system = MeanFieldPSpinSphericalSystem(N, p=p)
    run_gui(system)


def run_gui_db(dbname="pspin_spherical_p3_N20.sqlite"):
    from pele.gui import run_gui
    try:
        db = Database(dbname, createdb=False)
        interactions = db.get_property("interactions").value()
        nspins = db.get_property("nspins").value()
        p = db.get_property("p").value()
    except IOError:
        interactions=None
    system = MeanFieldPSpinSphericalSystem(nspins, p=p, interactions=interactions)
    run_gui(system, db=dbname)


if __name__ == "__main__":
    p = 3
    N = 150
    #run_gui(N, p)

    if False:
        system = MeanFieldPSpinSphericalSystem(N, p=p)
        db = system.create_database("pspin_spherical_p{}_N{}.sqlite".format(p,N))
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(20)

    if True:
        run_gui_db(dbname="pspin_spherical_p{}_N{}.sqlite".format(p,N))

    if False:
        compare_minima = lambda m1, m2 : compare_exact(m1.coords, m2.coords, rel_tol=1e-7, debug=False)
        db = Database("pspin_spherical_p{}_N{}.sqlite".format(p,N))
        minima = db.minima()
        minima.sort(key=lambda m: m.energy)
        #for m in minima:
        #    print m.energy, m.coords
        print minima[0].energy, minima[0].coords
        print minima[1].energy, minima[1].coords
        print compare_minima(minima[0],minima[1])

