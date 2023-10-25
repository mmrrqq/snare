from models.single_cls import SingleClassifier
from models.predicted_rotator import PredictiveRotator
from models.zero_shot_cls import ZeroShotClassifier
from models.rotator import Rotator

names = {
    # classifiers
    'single_cls': SingleClassifier,
    'zero_shot_cls': ZeroShotClassifier,

    # rotators
    'rotator': Rotator,
    'true_rotator': PredictiveRotator
}
