from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.sgsac import SGSAC
from algorithms.vre import VRE
from algorithms.dtk import DTK

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "vre": VRE,
    "dtk": DTK,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "sgsac": SGSAC,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
