from __future__ import division
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from pele.takestep import RandomCluster, RandomDisplacement
from pele.storage import Database
from learnscapes.NeuralNets import Elu3NNPotential
from learnscapes.NeuralNets import NNBaseSystem


class Elu3NNSystem(NNBaseSystem):
    def __init__(self, x_train_data, y_train_data, hnodes, hnodes2, reg=0, scale=0.1, dtype='float64', device='cpu'):
        super(Elu3NNSystem, self).__init__(x_train_data, y_train_data, scale=scale, dtype=dtype, device=device)
        self.hnodes = hnodes
        self.hnodes2 = hnodes2
        self.reg = reg
        self.ndim = (self.x_train_data.shape[1]*self.hnodes +
                     self.hnodes * self.hnodes2 +
                     self.hnodes2*self.y_train_data.shape[1] +
                     self.hnodes + self.hnodes2 +
                     self.y_train_data.shape[1])
        self.pot = self.get_potential(dtype=self.dtype, device=self.device)
        self.name = 'Elu3NN'

    def get_system_properties(self):
        return dict(potential="Elu3NNPotential",
                    x_train_data=self.x_train_data,
                    y_train_data=self.y_train_data,
                    reg=self.reg,
                    hnodes=self.hnodes,
                    hnodes2=self.hnodes2,
                    )

    def get_potential(self, dtype='float64', device='cpu'):
        try:
            return self.pot
        except AttributeError:
            self.pot = Elu3NNPotential(self.x_train_data, self.y_train_data, self.hnodes,
                                       self.hnodes2, reg=self.reg, dtype=dtype, device=device)
            return self.pot

    def get_takestep(self, **kwargs):
        """return the takestep object for use in basinhopping, etc."""
        return RandomDisplacement(stepsize=2.5*self.scale)

    # def get_compare_exact(self):
    #     """
    #     are they the same minima?
    #     """
    #     return lambda x1, x2 : compare_exact(x1, x2, rel_tol=1e-5, abs_tol=1e-7, debug=False)

def run_gui_db(dbname="regression_logit_mnist.sqlite", device='cpu'):
    from pele.gui import run_gui
    try:
        db = Database(dbname, createdb=False)
        x_train_data=db.get_property("x_train_data").value(),
        y_train_data=db.get_property("y_train_data").value(),
        hnodes=db.get_property("hnodes").value(),
        hnodes2=db.get_property("hnodes2").value(),
        reg=db.get_property("reg").value(),
    except IOError:
        pass
    hnodes, hnodes2, reg = hnodes[0], hnodes2[0], reg[0]
    x_train_data, y_train_data = np.array(np.array(x_train_data)[0,:,:]), np.array(np.array(y_train_data)[0,:,:])
    print np.array(x_train_data).shape, np.array(y_train_data).shape
    system = Elu3NNSystem(x_train_data, y_train_data, hnodes, hnodes2, reg=reg, device=device)
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
    hnodes2 = 10
    # system = Mlp3System(trX, trY, hnodes, reg=reg, device='cpu')
    system = Elu3NNSystem(trX, trY, hnodes, hnodes2, reg=reg, device='cpu')
    db = system.create_database("{}_mnist_h{}_h2{}_p{}_r{}.sqlite".format(system.name, hnodes, hnodes2, bs, reg))
    bh = True

    if bh:
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(10)
        # run_double_ended_connect(system, db, strategy='gmin')
        database_stats(system, db, teX, teY,
                       fname="{}_mnist_h{}_h2{}_p{}_r{}.sqlite".format(system.name, hnodes, hnodes2, bs, reg))

    if not bh:
        run_gui_db(dbname="{}_mnist_h{}_h2{}_p{}_r{}.sqlite".format(system.name, hnodes, hnodes2, bs, reg),
                   device='cpu')


if __name__ == "__main__":
    main()