from learnscapes.systems import  MeanFieldPSpinSphericalSystem
from pele.storage import Database

def create_system(nspins, p, interactions, dtype='float64', device='cpu'):
    system = MeanFieldPSpinSphericalSystem(nspins, p=p, interactions=interactions,
                                           dtype=dtype, device=device)
    return system

def get_database_params(dbname):
    db = Database(dbname, createdb=False)
    interactions = db.get_property("interactions").value()
    db_nspins = db.get_property("nspins").value()
    db_p = db.get_property("p").value()
    params = (db_nspins, db_p, interactions)
    return db, params

def get_database_params_worker(dbname, nspins, p):
    db, (db_nspins, db_p, interactions) = get_database_params(dbname)
    #close this SQLAlchemy session
    db.session.close()
    #check that parameters match
    assert db_nspins == nspins
    assert db_p == p
    return interactions

def get_database_params_server(dbname, nspins, p):
    db, (db_nspins, db_p, interactions) = get_database_params(dbname)
    #check that parameters match
    assert db_nspins == nspins
    assert db_p == p
    return db, interactions

def make_disconnectivity_graph(system, database, fname='dg.pdf', **kwargs):
    import matplotlib.pyplot as plt
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, **kwargs)
    dg.calculate()
    dg.plot(linewidth=1.5)
    plt.savefig(fname)