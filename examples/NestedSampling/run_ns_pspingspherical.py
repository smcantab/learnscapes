from __future__ import division
import tensorflow as tf
import argparse
from learnscapes.NestedSampling import NestedSampling
from learnscapes.pspinspherical import MeanFieldPSpinSphericalTF

# THIS REQUIRES A CUSTOM NESTED SAMPLING CLASS THAT IS NOT IMPLEMENTED YET

def main():
    parser = argparse.ArgumentParser(description="do nested sampling on the pspin glass model")
     # pspin variables
    parser.add_argument("p", type=int, help="p-spin")
    parser.add_argument("nspins", type=int, help="number of spins")
    parser.add_argument("--dtype", type=str, help="data type (recommended float64)", default='float64')
    parser.add_argument("--device", type=str, help="device on which TensorFlow should run", default='cpu')
    # nested sampling variables
    parser.add_argument("-K", "--nreplicas", type=int, help="number of replicas", default=300)
    parser.add_argument("-N", "--mciter", type=int, help="number of Monte Carlo steps for each Markov Chain",
                        default=100)
    parser.add_argument("-p", "--nproc", type=int, help="number of processors", default=1)
    parser.add_argument("-v", "--verbose", action="store_true", help="turn off verbose printing of "
                                                                     "information at every step")
    parser.add_argument("--label", type=str, help="label output file", default=None)
    parser.add_argument("--etol", type=float, help="scale for stepsize and random sampling", default=1e-8)

    args = parser.parse_args()
    print args
    # pspin parameters
    p, nspins = args.p, args.nspins
    dtype, device = args.dtype, args.device
    # NS paramters
    nreplicas, mciter = args.nreplicas, args.mciter
    nproc, verbose = args.nproc, args.verbose
    etol = args.etol
    if args.label is None:
        label = "pspin_spherical_p{}_N{}.ns".format(p,nspins)
    else:
        label = args.label

    norm = tf.random_normal([nspins for _ in xrange(p)], mean=0, stddev=1.0, dtype=dtype)
    interactions = norm.eval(session=tf.Session())

    pot = MeanFieldPSpinSphericalTF(interactions, nspins, p, dtype=dtype, device=device)
    # implementing nested sampling for the psping spherical model requires normalizing each step
    # I need to this this
    # ns = NestedSampling(pot, scale=scale, nreplicas=nreplicas, mciter=mciter, nproc=nproc, verbose=verbose)

if __name__ == "__main__":
    main()
