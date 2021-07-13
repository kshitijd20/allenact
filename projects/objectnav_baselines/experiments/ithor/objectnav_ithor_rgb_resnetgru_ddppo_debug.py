from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_base import (
    ObjectNaviThorBaseConfigDebug,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfigDebug,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)


class ObjectNaviThorRGBPPOExperimentConfig(
    ObjectNaviThorBaseConfigDebug, ObjectNavMixInPPOConfigDebug, ObjectNavMixInResNetGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNaviThorBaseConfigDebug.SCREEN_SIZE,
            width=ObjectNaviThorBaseConfigDebug.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=ObjectNaviThorBaseConfigDebug.TARGET_TYPES,),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-iTHOR-RGB-ResNetGRU-DDPPO-Debug"
