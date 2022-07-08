REGISTRY = {}

from .basic_controller import BasicMAC
from .latent_controller import LatentMAC
from .FuN_controller import FuNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["latent"] = LatentMAC
REGISTRY["FuN"] = FuNMAC
