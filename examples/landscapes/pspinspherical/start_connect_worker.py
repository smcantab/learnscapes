import argparse
from pele.concurrent import ConnectWorker
from examples.landscapes.pspinspherical import create_system, get_database_params_worker
from learnscapes.utils import get_server_uri

def main():
    parser = argparse.ArgumentParser(description="connect worker queue")
    parser.add_argument("p", type=int, help="p-spin")
    parser.add_argument("nspins", type=int, help="number of spins")
    parser.add_argument("--strategy", type=str, help="strategy to adopt: random (default), "
                                                     "untrap, combine, gmin", default="random")
    args = parser.parse_args()

    nspins = args.nspins
    p = args.p

    dbname = "pspin_spherical_p{}_N{}.sqlite".format(p,nspins)
    interactions = get_database_params_worker(dbname, nspins, p)
    system = create_system(nspins, p, interactions)

    fname = 'server_uri_pspin_spherical_p{}_N{}.uri'.format(p,nspins)
    uri = get_server_uri(fname, nspins, p)
    worker = ConnectWorker(uri, system=system, strategy=args.strategy)
    worker.run()


if __name__ == "__main__":
    main()
