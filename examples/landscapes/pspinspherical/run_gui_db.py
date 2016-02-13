from __future__ import division
import argparse
import os
from pele.gui import run_gui
from examples.landscapes.pspinspherical import create_system, get_database_params

def main():
    parser = argparse.ArgumentParser(description="do nested sampling on a 3 layer elu neural network")
    # psping variables
    parser.add_argument("dbname", type=str, help="database name")
    parser.add_argument("--dtype", type=str, help="data type (recommended float64)", default='float64')
    parser.add_argument("--device", type=str, help="device on which TensorFlow should run", default='cpu')


    args = parser.parse_args()
    print args
    # pspin parameters
    dbname = os.path.abspath(args.dbname)
    dtype, device = args.dtype, args.device

    db, (nspins, p, interactions) = get_database_params(dbname)
    system = create_system(nspins, p, interactions, dtype=dtype, device=device)

    run_gui(system, db)

if __name__ == "__main__":
    main()