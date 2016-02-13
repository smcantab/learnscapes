import argparse
from pele.concurrent import BasinhoppingWorker
from examples.landscapes.pspinspherical import create_system, get_database_params_worker
from learnscapes.utils import get_server_uri

def main():
    parser = argparse.ArgumentParser(description="connect worker queue")
    parser.add_argument("p", type=int, help="p-spin")
    parser.add_argument("nspins", type=int, help="number of spins")
    parser.add_argument("--nsteps", type=int, help="number of basin hopping steps", default=1000)
    args = parser.parse_args()

    nspins = args.nspins
    p = args.p

    dbname = "pspin_spherical_p{}_N{}.sqlite".format(p,nspins)
    interactions = get_database_params_worker(dbname, nspins, p)
    system = create_system(nspins, p, interactions)

    fname = 'server_uri_pspin_spherical_p{}_N{}.uri'.format(p,nspins)
    uri = get_server_uri(fname, nspins, p)
    worker = BasinhoppingWorker(uri, system=system)
    worker.run(args.nsteps)

if __name__ == "__main__":
    main()
