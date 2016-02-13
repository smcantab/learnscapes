from learnscapes.systems import Elu3NNSystem
from pele.storage import Database
from learnscapes.utils import isClose
import numpy as np

def create_system(x_train_data, y_train_data, hnodes, hnodes2,
                  reg=0, scale=1, dtype='float64', device='cpu'):
    system = Elu3NNSystem(x_train_data, y_train_data, hnodes, hnodes2,
                          reg=reg, scale=scale, dtype=dtype, device=device)
    return system

def get_database_params(dbname):
    db = Database(dbname, createdb=False)
    x_train_data = db.get_property("x_train_data").value()
    y_train_data = db.get_property("y_train_data").value()
    hnodes = db.get_property("hnodes").value()
    hnodes2 = db.get_property("hnodes2").value()
    reg = db.get_property("reg").value()
    params = (x_train_data, y_train_data, hnodes, hnodes2, reg)
    return db, params

def get_database_params_worker(dbname, hnodes, hnodes2, reg):
    db, (x_train_data, y_train_data, db_hnodes, db_hnodes2, db_reg) = get_database_params(dbname)
    #close this SQLAlchemy session
    db.session.close()
    #check that parameters match
    assert db_hnodes == hnodes
    assert db_hnodes2 == hnodes2
    assert isClose(db_reg, reg)
    return x_train_data, y_train_data

def get_database_params_server(dbname, hnodes, hnodes2, reg):
    db, (x_train_data, y_train_data, db_hnodes, db_hnodes2, db_reg) = get_database_params(dbname)
    #check that parameters match
    assert db_hnodes == hnodes
    assert db_hnodes2 == hnodes2
    assert isClose(db_reg, reg)
    return db, x_train_data, y_train_data