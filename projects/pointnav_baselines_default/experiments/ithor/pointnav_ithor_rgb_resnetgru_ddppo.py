from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from projects.pointnav_baselines_default.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines_default.experiments.pointnav_mixin_resnetgru import (
    PointNavMixInResnetGRUConfig,
)
from projects.pointnav_baselines_default.experiments.pointnav_thor_mixin_ddppo import (
    PointNavThorMixInPPOConfig,
)
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import Builder
class PointNaviThorRGBPPOExperimentConfig(
    PointNaviThorBaseConfig,
    PointNavThorMixInPPOConfig,
    PointNavMixInResnetGRUConfig,
):
    """An Point Navigation experiment configuration in iThor with RGB input."""

    PREPROCESSORS = [
        Builder(
            ResNetPreprocessor,
            {
                "input_height": PointNaviThorBaseConfig.SCREEN_SIZE,
                "input_width": PointNaviThorBaseConfig.SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "pool": False,
                "torchvision_resnet_model": models.resnet18,
                "input_uuids": ["rgb_lowres"],
                "output_uuid": "rgb_resnet",
            },
        ),
    ]

    SENSORS = [
        RGBSensorThor(
            height=PointNaviThorBaseConfig.SCREEN_SIZE,
            width=PointNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GPSCompassSensorRoboThor(),
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-RGB-Resnet-DDPPO"
