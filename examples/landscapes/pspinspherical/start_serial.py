from __future__ import division
import argparse
from examples.landscapes.pspinspherical import create_system, get_database_params_server, make_disconnectivity_graph
from learnscapes.utils import database_stats, run_double_ended_connect, refine_database

def main():
    parser = argparse.ArgumentParser(description="do nested sampling on a 3 layer elu neural network")
    # pspin variables
    parser.add_argument("p", type=int, help="p-spin")
    parser.add_argument("nspins", type=int, help="number of spins")
    parser.add_argument("--dtype", type=str, help="data type (recommended float32)", default='float32')
    parser.add_argument("--device", type=str, help="device on which TensorFlow should run", default='cpu')
    # operations to perform
    parser.add_argument("--bh", type=int, help="number of basin hopping steps to perform", default=0)
    parser.add_argument("--connect", action="store_true", help="run all")
    parser.add_argument("--connect-method", type=str, help="method used to connect", default='random')


    args = parser.parse_args()
    print args
    # pspin parameters
    p, nspins = args.p, args.nspins
    dtype, device = args.dtype, args.device
    #operations
    bh_niter = args.bh
    connect, connect_method = args.connect, args.connect_method
    dbname = "pspin_spherical_p{}_N{}.sqlite".format(p,nspins)
    try:
        db, interactions = get_database_params_server(dbname, nspins, p)
        print "Warning: database {} already exists, using the already existing database".format(dbname)
    except IOError:
        db = None
        interactions = None

    system = create_system(nspins, p, interactions, dtype=dtype, device=device)

    if db is None:
        db = system.create_database(dbname)

    #now actually run the computattion
    fname = None
    if bh_niter > 0:
        bh = system.get_basinhopping(database=db, outstream=None)
        bh.run(bh_niter)
        print "\n refining database \n"
        refine_database(system, db, tol=1e-10, nsteps=int(2.5e4), maxstep=1, iprint=1000)
    if connect:
        fname = "pspin_spherical_p{}_N{}.dg.pdf".format(p,nspins)
        run_double_ended_connect(system, db, strategy=connect_method)

    if fname is not None:
        make_disconnectivity_graph(system, db, fname=fname)


if __name__ == "__main__":
    main()