from __future__ import division
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from learnscapes.NestedSampling import NestedSampling
from learnscapes.NeuralNets import Elu3NNPotential

def main():
    parser = argparse.ArgumentParser(description="do nested sampling on a 3 layer elu neural network")
    # nested sampling variables
    parser.add_argument("-K", "--nreplicas", type=int, help="number of replicas", default=300)
    parser.add_argument("-N", "--mciter", type=int, help="number of Monte Carlo steps for each Markov Chain",
                        default=100)
    parser.add_argument("-p", "--nproc", type=int, help="number of processors", default=1)
    parser.add_argument("-v", "--verbose", action="store_true", help="turn off verbose printing of "
                                                                     "information at every step")
    parser.add_argument("--label", type=str, help="label output file", default=None)
    parser.add_argument("--etol", type=float, help="scale for stepsize and random sampling", default=1e-8)
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

    args = parser.parse_args()
    print args
    # NN parameters
    model, ntrain = args.model.lower(), args.ntrain
    reg, scale = args.l2reg, args.scale
    hnodes, hnodes2 = args.hnodes, args.hnodes2
    dtype, device = args.dtype, args.device
    # NS paramters
    nreplicas, mciter = args.nreplicas, args.mciter
    nproc, verbose = args.nproc, args.verbose
    etol = args.etol
    if args.label is None:
        label = "{}_mnist_h{}_h2{}_p{}_r{}.ns".format('elu3NN', hnodes, hnodes2, ntrain, reg)
    else:
        label = args.label

    if model == 'mnist':
        data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    dsize = len(data.train.images)
    trX, trY = data.train.images[::int(dsize/ntrain)], data.train.labels[::int(dsize/ntrain)]
    teX, teY = data.test.images, data.test.labels

    pot = Elu3NNPotential(trX, trY, hnodes, hnodes2, reg=reg, dtype=dtype, device=device)
    ns = NestedSampling(pot, scale=scale, nreplicas=nreplicas, mciter=mciter, nproc=nproc, verbose=verbose)

    #now actually run the computattion
    ns.run(label=label, etol=etol, maxiter=None, iprint_replicas=1000)
    lowest_replica = ns.ns.replicas[-1]
    print pot.test_model(lowest_replica.x, teX, teY)

if __name__ == "__main__":
    main()
