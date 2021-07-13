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
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_rgb_resnetgru_ddppo_debug import (
    ObjectNaviThorRGBPPOExperimentConfig,
)
from allenact.utils.viz_utils import (
    VizSuite,
    TrajectoryViz,
    AgentViewViz,
    ActorViz,
    TensorViz1D,
)
from allenact_plugins.ithor_plugin.ithor_viz import ThorViz


class ObjectNaviThorRGBPPOVizExperimentConfig(
    ObjectNaviThorRGBPPOExperimentConfig
):
    """A simple object navigation experiment in THOR.

    Training with DAgger and then PPO + using viz for test.
    """

    TEST_SAMPLES_IN_SCENE = 4

    @classmethod
    def tag(cls):
        return "ObjectNavThorDaggerThenPPOViz"

    viz = None

    def get_viz(self, mode):
        if self.viz is not None:
            return self.viz

        self.viz = VizSuite(
            mode=mode,
            base_trajectory=TrajectoryViz(
                path_to_target_location=None, path_to_rot_degrees=("rotation",),
            ),
            egeocentric=AgentViewViz(max_video_length=100),
            action_probs=ActorViz(figsize=(3.25, 10), fontsize=18),
            taken_action_logprobs=TensorViz1D(),
            episode_mask=TensorViz1D(rollout_source=("masks",)),
            thor_trajectory=ThorViz(
                path_to_target_location=None, figsize=(8, 8), viz_rows_cols=(448, 448),
            ),
        )

        return self.viz

    def machine_params(self, mode="train", **kwargs):
        params = super().machine_params(mode, **kwargs)

        if mode == "test":
            params.set_visualizer(self.get_viz(mode))

        return params





