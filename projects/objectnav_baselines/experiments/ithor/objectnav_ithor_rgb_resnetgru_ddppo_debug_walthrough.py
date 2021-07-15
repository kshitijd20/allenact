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
from typing import (
    Optional,
    Any,
    Dict,
    Union,
    List,
    Sequence,
    cast,
    Iterator,
    Callable,
)
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_rgb_resnetgru_ddppo_debug import (
    ObjectNaviThorRGBPPOExperimentConfig,
)
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from collections import defaultdict
from typing import Dict, List, Tuple, cast
from allenact.utils.experiment_utils import set_seed
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.utils.tensor_utils import batch_observations
from allenact.base_abstractions.misc import Memory, ActorCriticOutput
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor
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
DEFAULT_VALID_GPU_IDS = (torch.cuda.device_count() - 1,)
def get_model(config):
    mode = "valid"

    nprocesses = 1
    devices = (
        [torch.device("cpu")]
        if not torch.cuda.is_available()
        else DEFAULT_VALID_GPU_IDS
    )


    sensors = [*config.SENSORS]
    if mode != "train":
        sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

    sensor_preprocessor_graph = (
        SensorPreprocessorGraph(
            source_observation_spaces=SensorSuite(sensors).observation_spaces,
            preprocessors=config.preprocessors(),
        )
        if mode == "train"
           or (
                   (isinstance(nprocesses, int) and nprocesses > 0)
                   or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
           )
        else None
    )


    machine_params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=DEFAULT_VALID_GPU_IDS
            if mode == "train"
            else devices,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )
    machine_params_self: MachineParams
    worker_id = 0
    device = "cpu"
    if isinstance(machine_params, MachineParams):
        machine_params_self = machine_params
    else:
        machine_params_self = MachineParams(**machine_params)

    num_samplers_per_worker = machine_params_self.nprocesses
    num_samplers = num_samplers_per_worker[worker_id]
    sensor_preprocessor_graph = None
    actor_critic: Optional[ActorCriticModel] = None
    if num_samplers > 0:
        create_model_kwargs = {}
        if machine_params.sensor_preprocessor_graph is not None:
            sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph.to(
                device
            )
            create_model_kwargs[
                "sensor_preprocessor_graph"
            ] = sensor_preprocessor_graph
        actor_critic = cast(
            ActorCriticModel, config.create_model(**create_model_kwargs),
        ).to(device)

    return actor_critic,sensor_preprocessor_graph
def test_pretrained_objectnav_walkthrough_mapping_agent( tmpdir):
    devices = (
        [torch.device("cpu")]
        if not torch.cuda.is_available()
        else DEFAULT_VALID_GPU_IDS
    )
    print("Creating sampler")
    exp_config = ObjectNaviThorRGBPPOExperimentConfig()
    sampler_args = exp_config.valid_task_sampler_args(process_ind = 0,total_processes=1,devices=devices)

    walkthrough_task_sampler = exp_config.make_sampler_fn(**sampler_args)
    print("Created sampler")
    print("------------------------------------------------------------------------")

    ckpt_path = os.path.join(
        tmpdir, "exp_Objectnav-iTHOR-RGB-ResNetGRU-DDPPO-Debug__stage_00__steps_000000163840.pt"
    )
    print("Loading checkpoint")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    walkthrough_model,sensor_preprocessor_graph = get_model(ObjectNaviThorRGBPPOExperimentConfig)
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
    #return 0
    observations_dict = defaultdict(lambda: [])
    for i in range(3):
        masks = 0 * masks

        set_seed(i + 1)
        task = walkthrough_task_sampler.next_task()

        obs_list = observations_dict[i]
        obs_list.append(task.get_observations())

        print("Task is ", task)
        def add_step_dim(input):
            if isinstance(input, torch.Tensor):
                return input.unsqueeze(0)
            elif isinstance(input, Dict):
                return {k: add_step_dim(v) for k, v in input.items()}
            else:
                raise NotImplementedError

        #batch = add_step_dim(batch_observations([task.get_observations()]))
        batch = add_step_dim(sensor_preprocessor_graph.get_observations(batch_observations([task.get_observations()])))
        print("Batch is ", batch)
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

            obs_list.append(obs)
            #batch = add_step_dim(batch_observations([obs]))
            batch = add_step_dim(sensor_preprocessor_graph.get_observations(batch_observations([task.get_observations()])))
            if task.num_steps_taken() >= 10:
                break
    os.makedirs("tmp_out", exist_ok=True)
    compress_pickle.dump(
         {**observations_dict}, "tmp_out/rearrange_mapping_examples.pkl.gz"
    )
    try:
        walkthrough_task_sampler.close()
    except NameError:
        pass

tmpdir = "/home/kshitijd/projects/allenact/storage/objectnav-ithor-rgbdebug/checkpoints/Objectnav-iTHOR-RGB-ResNetGRU-DDPPO-Debug/2021-07-13_02-20-39"
test_pretrained_objectnav_walkthrough_mapping_agent(tmpdir)
