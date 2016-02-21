from linear_algebra import tfDot, tfNorm, tfRnorm, tfOrthog, tfNnorm, reduce_mean, l2_loss
from utils import select_device_simple, isClose, isCloseArray, mindist_1d
from utils import database_stats, run_double_ended_connect, make_disconnectivity_graph, refine_database
from pyro_utils import get_server_uri, pick_unused_port, write_server_uri