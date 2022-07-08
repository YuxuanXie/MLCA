REGISTRY = {}

from .rnn_agent import RNNAgent
from .FuN_agent import FuNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["FuN"] = FuNAgent