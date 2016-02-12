from __future__ import division
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from pele.takestep import RandomCluster, RandomDisplacement
from pele.storage import Database
from learnscapes.NeuralNets import DoubleLogisticRegressionPotential, Elu2NNPotential
from learnscapes.NeuralNets import NNBaseSystem
from learnscapes.utils import mindist_1d


class Mlp3System(NNBaseSystem):
    def __init__(self, x_train_data, y_train_data, hnodes, reg=0.1, dtype='float64', device='cpu'):
        super(Mlp3System, self).__init__(x_train_data, y_train_data, dtype=dtype, device=device)
        self.hnodes = hnodes
        self.reg = reg
        self.ndim = (self.y_train_data.shape[1]*self.hnodes + self.hnodes*self.x_train_data.shape[1] + self.hnodes + self.y_train_data.shape[1])
        self.pot = self.get_potential(dtype=self.dtype, device=self.device)
        self.name = 'Mlp3'

    def get_system_properties(self):
        return dict(potential="Mlp3Potential",
                    x_train_data=self.x_train_data,
                    y_train_data=self.y_train_data,
                    reg=self.reg,
                    hnodes=self.hnodes,
                    )

    def get_potential(self, dtype='float64', device='cpu'):
        try:
            return self.pot
        except AttributeError:
            self.pot = DoubleLogisticRegressionPotential(self.x_train_data, self.y_train_data, self.hnodes,
                                                         reg=self.reg, dtype=dtype, device=device)
            return self.pot
    
    def get_mindist(self):
        return lambda x1, x2 : mindist_1d(x1, x2)

    def get_random_configuration(self):
        return np.random.normal(0, scale=1, size=self.ndim)

    def get_takestep(self, **kwargs):
        """return the takestep object for use in basinhopping, etc."""
        return RandomDisplacement(stepsize=2.5)

    # def get_compare_exact(self):
    #     """
    #     are they the same minima?
    #     """
    #     return lambda x1, x2 : compare_exact(x1, x2, rel_tol=1e-5, abs_tol=1e-7, debug=False)

class Elu2NNSystem(Mlp3System):
    def __init__(self, x_train_data, y_train_data, hnodes, reg=0.1, dtype='float64', device='cpu'):
        super(Elu2NNSystem, self).__init__(x_train_data, y_train_data, hnodes,
                                           reg=reg, dtype=dtype, device=device)
        self.name = 'Elu2NN'

    def get_system_properties(self):
        return dict(potential="Elu2NNPotential",
                    x_train_data=self.x_train_data,
                    y_train_data=self.y_train_data,
                    reg=self.reg,
                    hnodes=self.hnodes,
                    )

    def get_potential(self, dtype='float64', device='cpu'):
        try:
            return self.pot
        except AttributeError:
            self.pot = Elu2NNPotential(self.x_train_data, self.y_train_data, self.hnodes,
                                                         reg=self.reg, dtype=dtype, device=device)
            return self.pot

def run_gui_db(dbname="regression_logit_mnist.sqlite", device='cpu'):
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
    # system = Mlp3System(x_train_data, y_train_data, hnodes, reg=reg, device=device)
    system = Elu2NNSystem(x_train_data, y_train_data, hnodes, reg=reg, device=device)
    run_gui(system, db=dbname)
    # from pele.thermodynamics import get_thermodynamic_information
    # get_thermodynamic_informationormation(system, db, nproc=1, verbose=True)

def main():
    from learnscapes.utils import database_stats, make_disconnectivity_graph, run_double_ended_connect
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = 1000
    trX, trY, teX, teY = mnist.train.images[::int(50000/bs)], mnist.train.labels[::int(50000/bs)], mnist.test.images, mnist.test.labels
    reg=0
    hnodes = 100
    # system = Mlp3System(trX, trY, hnodes, reg=reg, device='cpu')
    system = Elu2NNSystem(trX, trY, hnodes, reg=reg, device='cpu')
    db = system.create_database("{}_mnist_h{}_p{}_r{}.sqlite".format(system.name, hnodes, bs, reg))
    bh = True

    if bh:
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(10)
        # run_double_ended_connect(system, db, strategy='gmin')
        database_stats(system, db, teX, teY, fname="{}_mnist_h{}_p{}_r{}.sqlite".format(system.name, hnodes, bs, reg))

    if not bh:
        run_gui_db(dbname="{}_mnist_h{}_p{}_r{}.sqlite".format(system.name, hnodes, bs, reg), device='cpu')

    # if bh:
    #     # compare_minima = lambda m1, m2 : compare_exact(np.sort(np.abs(m1.coords)), np.sort(np.abs(m2.coords)), rel_tol=1e-5, debug=False)
    #     db = Database("mlp3_mnist_h{}_p{}_r{}.sqlite".format(hnodes, bs, reg))
    #     minima = db.minima()
    #     minima.sort(key=lambda m: m.energy)
    #     for m in minima:
    #        print m.energy#, m.coords
    #     # print minima[0].energy, np.sort(np.abs(minima[0].coords))
    #     # print minima[1].energy, np.sort(np.abs(minima[1].coords))
    #     # print compare_minima(minima[2],minima[3])


if __name__ == "__main__":
    main()