from learnscapes.systems import  MeanFieldPSpinSphericalSystem
from pele.storage import Database

def create_system(nspins, p, interactions, dtype='float32', device='gpu'):
    system = MeanFieldPSpinSphericalSystem(nspins, p=p, interactions=interactions,
                                           dtype=dtype, device=device)
    return system

def _get_database_params(dbname):
    db = Database(dbname, createdb=False)
    interactions = db.get_property("interactions").value()
    db_nspins = db.get_property("nspins").value()
    db_p = db.get_property("p").value()
    params = (db_nspins, db_p, interactions)
    return db, params

def get_database_params_worker(dbname, nspins, p):
    db, (db_nspins, db_p, interactions) = _get_database_params(dbname)
    #close this SQLAlchemy session
    db.session.close()
    #check that parameters match
    assert db_nspins == nspins
    assert db_p == p
    return interactions

def get_database_params_server(dbname, nspins, p):
    db, (db_nspins, db_p, interactions) = _get_database_params(dbname)
    #check that parameters match
    assert db_nspins == nspins
    assert db_p == p
    return db, interactions