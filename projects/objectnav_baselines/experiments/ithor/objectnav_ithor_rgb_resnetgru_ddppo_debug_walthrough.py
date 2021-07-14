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
from typing import Dict, List, Tuple, cast
from allenact.utils.experiment_utils import set_seed
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.utils.tensor_utils import batch_observations
from allenact.base_abstractions.misc import Memory, ActorCriticOutput

import compress_pickle
import numpy as np

from allenact_plugins.ithor_plugin.ithor_sensors import (
    RelativePositionChangeTHORSensor,
    ReachableBoundsTHORSensor,
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
import os
import torch

def test_pretrained_objectnav_walkthrough_mapping_agent( tmpdir):

    print("Creating sampler")
    sampler_args = ObjectNaviThorRGBPPOExperimentConfig.valid_task_sampler_args(process_ind = 0,total_processes=1)

    walkthrough_task_sampler = ObjectNaviThorRGBPPOExperimentConfig.make_sampler_fn(sampler_args)
    print("Created sampler")
    print("------------------------------------------------------------------------")

    ckpt_path = os.path.join(
        tmpdir, "exp_Objectnav-iTHOR-RGB-ResNetGRU-DDPPO-Debug__stage_00__steps_000000163840.pt"
    )
    print("Loading checkpoint")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    walkthrough_model = ObjectNaviThorRGBPPOExperimentConfig.create_model()
    walkthrough_model.load_state_dict(state_dict["model_state_dict"])

    print("Loaded checkpoint")
    print("------------------------------------------------------------------------")

    print("Creating rollout storage")
    rollout_storage = RolloutStorage(
        num_steps=1,
        num_samplers=1,
        actor_critic=walkthrough_model,
        only_store_first_and_last_in_memory=False,
    )
    print("Created rollout storage")
    print("------------------------------------------------------------------------")
    memory = rollout_storage.pick_memory_step(0)
    masks = rollout_storage.masks[:1]
    print("Memory is ", memory)
    try:
        walkthrough_task_sampler.close()
    except NameError:
        pass
    return 0
    for i in range(5):
        masks = 0 * masks

        set_seed(i + 1)
        task = walkthrough_task_sampler.next_task()
        print("Task is ", task)
        def add_step_dim(input):
            if isinstance(input, torch.Tensor):
                return input.unsqueeze(0)
            elif isinstance(input, Dict):
                return {k: add_step_dim(v) for k, v in input.items()}
            else:
                raise NotImplementedError

        batch = add_step_dim(batch_observations([task.get_observations()]))

        while not task.is_done():
            ac_out, memory = cast(
                Tuple[ActorCriticOutput, Memory],
                walkthrough_model.forward(
                    observations=batch,
                    memory=memory,
                    prev_actions=None,
                    masks=masks,
                ),
            )


            masks = masks.fill_(1.0)
            obs = task.step(
                action=ac_out.distributions.sample().item()
            ).observation
            batch = add_step_dim(batch_observations([obs]))

            if task.num_steps_taken() >= 10:
                break
    #os.makedirs("tmp_out", exist_ok=True)
    #compress_pickle.dump(
    #     {**observations_dict}, "tmp_out/rearrange_mapping_examples.pkl.gz"
    #)
    try:
        walkthrough_task_sampler.close()
    except NameError:
        pass

tmpdir = "/home/kshitijd/projects/allenact/storage/objectnav-ithor-rgbdebug/checkpoints/Objectnav-iTHOR-RGB-ResNetGRU-DDPPO-Debug/2021-07-13_02-20-39"
test_pretrained_objectnav_walkthrough_mapping_agent(tmpdir)