import socket

def pick_unused_port():
    """
    pick an unused port number

    Returns
    -------
    port: int
        free port number
    Notes
    -----
    this does not guarantee that the port is actually free, in the short time window
    between when the port is identified as free and it is actually bound to, the port
    can be taken by another process
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    addr, port = s.getsockname()
    s.close()
    return port

def get_server_uri(fname, nspins, p):
    with open(fname) as f:
        uri = [line for line in f][0]
    assert uri[:5] == "PYRO:"
    return uri

def write_server_uri(fname, server_name, hostname, port, nspins, p):
    uri = "PYRO:%s@%s:%d" % (server_name, hostname, port)
    with open(fname,'w') as out_server_uri:
        out_server_uri.write(uri)
    return uri