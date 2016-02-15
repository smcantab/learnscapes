from __future__ import division
import argparse
from examples.landscapes.NeuralNets.elu3NN import create_system
from learnscapes.utils import database_stats, run_double_ended_connect, refine_database

def main():
    parser = argparse.ArgumentParser(description="do nested sampling on a 3 layer elu neural network")
    # NN variables
    parser.add_argument("--model", type=str, help="model on which to train the NN: "
                        "(1) MNIST", default='mnist')
    parser.add_argument("--ntrain", type=int, help="number of training input nodes", default=1000)
    parser.add_argument("--hnodes", type=int, help="number of hidden nodes in first layer", default=10)
    parser.add_argument("--hnodes2", type=int, help="number of hidden nodes in second layer", default=10)
    parser.add_argument("--l2reg", type=float, help="l2 regularization constant", default=0)
    parser.add_argument("--scale", type=float, help="scale for stepsize and random sampling", default=1)
    parser.add_argument("--dtype", type=str, help="data type (recommended float64)", default='float64')
    parser.add_argument("--device", type=str, help="device on which TensorFlow should run", default='cpu')
    # operations to perform
    parser.add_argument("--bh", type=int, help="number of basin hopping steps to perform", default=0)
    parser.add_argument("--connect", action="store_true", help="run all")
    parser.add_argument("--connect-method", type=str, help="method used to connect", default='random')


    args = parser.parse_args()
    print args
    # NN parameters
    model, ntrain = args.model.lower(), args.ntrain
    reg, scale = args.l2reg, args.scale
    hnodes, hnodes2 = args.hnodes, args.hnodes2
    dtype, device = args.dtype, args.device
    #operations
    bh_niter = args.bh
    connect, connect_method = args.connect, args.connect_method

    if model == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    else:
        raise NotImplementedError

    dsize = len(data.train.images)
    trX, trY = data.train.images[::int(dsize/ntrain)], data.train.labels[::int(dsize/ntrain)]
    teX, teY = data.test.images, data.test.labels

    system = create_system(trX, trY, hnodes, hnodes2, reg=reg, scale=scale, dtype=dtype, device=device)
    db = system.create_database("{}_{}_h{}_h2{}_p{}_r{}.sqlite".format(system.name, model, hnodes, hnodes2, ntrain, reg))

    #now actually run the computattion
    fname = None
    if bh_niter > 0:
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(bh_niter)
        print "\n refining database \n"
        refine_database(system, db, tol=1e-11, nsteps=int(1e5), maxstep=1, iprint=1000)

    if connect:
        fname = "{}_{}_h{}_h2{}_p{}_r{}.dg.pdf".format(system.name, model, hnodes, hnodes2, ntrain, reg)
        run_double_ended_connect(system, db, strategy=connect_method)

    database_stats(system, db, teX, teY, fname=fname)


if __name__ == "__main__":
    main()