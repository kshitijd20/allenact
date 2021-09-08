from abc import ABC
from typing import Sequence, Union

import gym
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import Builder

# fmt: off
try:
    # Habitat may not be installed, just create a fake class here in that case
    from allenact_plugins.habitat_plugin.habitat_sensors import TargetCoordinatesSensorHabitat
except ImportError:
    class TargetCoordinatesSensorHabitat:  #type:ignore
        pass
# fmt: on

from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavActorCriticSimpleConvRNN,
    ResnetTensorPointNavActorCritic
)


class PointNavMixInResnetGRUConfig(PointNavBaseConfig, ABC):
    """The base config for all iTHOR PPO PointNav experiments."""
    
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        goal_sensor_uuid = next(
            (
                s.uuid
                for s in cls.SENSORS
                if isinstance(
                    s, (GPSCompassSensorRoboThor, TargetCoordinatesSensorHabitat)
                )
            ),
            None,
        )
        
        return ResnetTensorPointNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
        )
        