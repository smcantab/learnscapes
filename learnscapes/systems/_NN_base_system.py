from __future__ import division
import numpy as np
from pele.systems import BaseSystem
from pele.systems.basesystem import dict_copy_update
from pele.optimize import lbfgs_cpp
from pele.landscape import smooth_path
from pele.takestep import RandomDisplacement
from learnscapes.utils import mindist_1d


class NNBaseSystem(BaseSystem):
    def __init__(self, x_train_data, y_train_data, scale=1, dtype='float64', device='cpu'):
        BaseSystem.__init__(self)
        self.dtype = dtype
        self.device = device
        self.x_train_data = np.array(x_train_data, dtype=dtype)
        self.y_train_data = np.array(y_train_data, dtype=dtype)
        self.setup_params(self.params)
        self.scale = scale

    def setup_params(self, params):
        params.takestep.verbose = True
        lcp = params.double_ended_connect.local_connect_params
        lcp.pushoff_params.update(dict(stepmin=0.5))
        lcp.tsSearchParams.lowestEigenvectorQuenchParams.update(dict(dx=1e-3, first_order=False))

        nebparams = lcp.NEBparams
        # nebparams.image_density = 0.8
        # nebparams.iter_density = 50.
        # nebparams.reinterpolate = 50
        nebparams.adaptive_nimages = True
        nebparams.adaptive_niter = True
        nebparams.adjustk_freq = 10
        nebparams.k = 2000

        params.structural_quench_params.tol = 1e-7
        params.structural_quench_params.maxstep = 1
        params.structural_quench_params.M = 4
        params.structural_quench_params.iprint = 1000
        params.structural_quench_params.verbosity = 0

        params.database.overwrite_properties = False

        # params.basinhopping.insert_rejected = True
        params.basinhopping.temperature = 10000

        lcp.tsSearchParams.hessian_diagonalization = False

        params.database.accuracy = 1e-5

    def get_mindist(self):
        return lambda x1, x2 : mindist_1d(x1, x2)

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

    def get_takestep(self, **kwargs):
        """return the takestep object for use in basinhopping, etc."""
        return RandomDisplacement(stepsize=self.scale)

    def get_random_configuration(self):
        return np.random.normal(0, scale=self.scale, size=self.ndim)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None

    def get_metric_tensor(self, coords):
        return None

    def smooth_path(self, path, **kwargs):
        mindist = self.get_mindist()
        return smooth_path(path, mindist, **kwargs)

    def create_database(self, *args, **kwargs):
        return BaseSystem.create_database(self, *args, **kwargs)

    def draw(self, coords, index):
        pass

    def get_metric_tensor(self, coords):
        return None

    def get_nzero_modes(self):
        return 0

    def get_pgorder(self, coords):
        return 1