from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .coma_entropy_learner import COMAEntropyLearner
from .latent_learner import LatentLearner
from .FuN_learner import FuNLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_entropy_learner"] = COMAEntropyLearner
REGISTRY["latent_learner"] = LatentLearner 
REGISTRY["FuN_learner"] = FuNLearner 

