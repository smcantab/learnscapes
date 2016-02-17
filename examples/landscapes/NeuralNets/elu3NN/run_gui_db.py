from __future__ import division
import argparse
import os
from pele.gui import run_gui
from examples.landscapes.NeuralNets.elu3NN import create_system, get_database_params
from learnscapes.utils import database_stats, run_double_ended_connect

def main():
    parser = argparse.ArgumentParser(description="do nested sampling on a 3 layer elu neural network")
    # NN variables
    parser.add_argument("dbname", type=str, help="database name")
    parser.add_argument("--scale", type=float, help="scale for stepsize and random sampling", default=1)
    parser.add_argument("--dtype", type=str, help="data type (recommended float32)", default='float32')
    parser.add_argument("--device", type=str, help="device on which TensorFlow should run", default='cpu')


    args = parser.parse_args()
    print args
    # NN parameters
    dbname = os.path.abspath(args.dbname)
    scale = args.scale
    dtype, device = args.dtype, args.device

    db, (trX, trY, hnodes, hnodes2, reg) = get_database_params(dbname)
    system = create_system(trX, trY, hnodes, hnodes2, reg=reg, scale=scale, dtype=dtype, device=device)

    run_gui(system, db)

if __name__ == "__main__":
    main()