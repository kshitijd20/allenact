import os
import platform
import random
import sys
import urllib
import urllib.request
import warnings
import copy
from collections import defaultdict

# noinspection PyUnresolvedReferences
from tempfile import mkdtemp
from typing import Dict, List, Tuple, cast

# noinspection PyUnresolvedReferences
import ai2thor

# noinspection PyUnresolvedReferences
import ai2thor.wsgi_server
import compress_pickle
import numpy as np
import torch
import glob

from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.base_abstractions.misc import Memory, ActorCriticOutput
from allenact.embodiedai.mapping.mapping_utils.map_builders import SemanticMapBuilder
from allenact.utils.experiment_utils import set_seed
from allenact.utils.system import get_logger
from allenact.utils.tensor_utils import batch_observations
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RelativePositionChangeTHORSensor,
    ReachableBoundsTHORSensor,
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.tutorials.object_nav_ithor_ppo_baseline import ObjectNavThorPPOExperimentConfig
from math import ceil
from typing import Dict, Any, List, Optional, Sequence
import glob
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
from tqdm import tqdm

from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph

from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import Builder
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.ithor_plugin.ithor_task_samplers import (
    ObjectNaviThorDatasetTaskSampler,
)
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
import json

allenact_to_ai2thor_actions = {
    "MoveAhead" : "MoveAhead",
    "RotateRight" : "RotateRight",
    "RotateLeft" : "RotateLeft",
    "LookUp" : "LookUp",
    "LookDown" : "LookDown",
    "End" : "Stop"
} 


def read_metric_file(metric_file):
    with open(metric_file, "r") as read_file:
        allenact_val_metrics = json.load(read_file)
    challenge_metrics = {"episodes" : {}}
    for episode in allenact_val_metrics:
        episode_metrics = {}
        episode_metrics["rnn"] = episode["rnn"]
        #print(episode["followed_path"])
        episode_metrics["trajectory"] = [{
            "x" : p["x"],
            "y" : p["y"],
            "z" : p["z"],
            "rotation" : p["rotation"]["y"],
            "horizon" : p["horizon"]
        } for p in episode["followed_path"]]

        episode_metrics["actions_taken"] = [{
            "action": allenact_to_ai2thor_actions[a]
        } for a in episode["taken_actions"]]

        if episode_metrics["actions_taken"][-1] == {"action" : "Stop"}:
            episode_metrics["trajectory"].append(
                episode_metrics["trajectory"][-1]
            )

        """
        for i in range(len(episode_metrics["actions_taken"])):
            if episode_metrics["actions_taken"][i]["action"] == "Stop":
                action_success = episode["success"] 
            else:
                prev_traj = episode_metrics["trajectory"][i]
                next_traj = episode_metrics["trajectory"][i+1]
                action_success = prev_traj != next_traj
            episode_metrics["actions_taken"][i]["success"] = action_success
        """
        #challenge_metrics["success"] = sum([e["success"] for e in challenge_metrics["episodes"].values()]) / num_episodes
        #challenge_metrics["spl"] = ai2thor.util.metrics.compute_spl(episode_results)
        challenge_metrics["episodes"][episode["id"]] = episode_metrics

    num_episodes = len(challenge_metrics["episodes"])

    challenge_metrics["ep_len"] = sum([len(e["trajectory"]) for e in challenge_metrics["episodes"].values()]) / num_episodes

    print("number of episodes ", num_episodes, "average episode length ", challenge_metrics["ep_len"] )

def random_objectnav_metadata_extraction():
    open_x_displays = []
    try:
        open_x_displays = get_open_x_displays()
    except (AssertionError, IOError):
        pass


    # Define Task sampler 
    objectnav_task_sampler = (
        ObjectNavThorPPOExperimentConfig.make_sampler_fn(
            loop_dataset = False,
            mode="valid",
            seed=12,
            x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            scenes=ObjectNavThorPPOExperimentConfig.TRAIN_SCENES,
            scene_directory = os.path.join(os.getcwd(), "datasets/ithor-objectnav/train/episodes"),
            object_types= ObjectNavThorPPOExperimentConfig.OBJECT_TYPES,
            env_args= ObjectNavThorPPOExperimentConfig.ENV_ARGS,
            max_steps= ObjectNavThorPPOExperimentConfig.MAX_STEPS,
            sensors= ObjectNavThorPPOExperimentConfig.SENSORS,
            action_space= gym.spaces.Discrete(
                len(ObjectNaviThorGridTask.class_action_names())
            ),
            rewards_config = ObjectNavThorPPOExperimentConfig.REWARD_CONFIG,
        )
    )

    # Define model and load state dict
    tmpdir = "/home/ubuntu/projects/allenact/storage_default/objectnav_ithor_baseline_rs30/checkpoints/ObjectNaviThorPPOResnetGRU/2021-09-01_00-52-40"
    ckpt_string = "exp_ObjectNaviThorPPOResnetGRU__stage_00__steps_000050025710"
    ckpt_path = os.path.join(
            tmpdir, ckpt_string + ".pt"
        )
 
    state_dict = torch.load(
        ckpt_path,
        map_location="cpu",
    )
    mode='valid'
    nprocesses = 1
    sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(ObjectNavThorPPOExperimentConfig.SENSORS).observation_spaces,
                preprocessors=ObjectNavThorPPOExperimentConfig.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )
    walkthrough_model = ObjectNavThorPPOExperimentConfig.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph)
    random_model = ObjectNavThorPPOExperimentConfig.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph)
    walkthrough_model.load_state_dict(state_dict["model_state_dict"])
    print("Model Loaded")

    # 

    rollout_storage = RolloutStorage(
            num_steps=1,
            num_samplers=1,
            actor_critic=walkthrough_model,
            only_store_first_and_last_in_memory=False,
        )
    memory = rollout_storage.pick_memory_step(0)
    masks = rollout_storage.masks[:1]


    random_rollout_storage = RolloutStorage(
            num_steps=1,
            num_samplers=1,
            actor_critic=random_model,
            only_store_first_and_last_in_memory=False,
        )
    random_memory = random_rollout_storage.pick_memory_step(0)
    random_masks = random_rollout_storage.masks[:1]

    binned_map_losses = []
    semantic_map_losses = []
    num_tasks = 1000
    count = 0
    success_count =0 
    task_metrics = []
    random_task_metrics = []
    for i in tqdm(range(num_tasks)):
        masks = 0 * masks
        random_masks = 0 * random_masks
        task = objectnav_task_sampler.next_task()
        print((objectnav_task_sampler.max_tasks))
        if (objectnav_task_sampler.max_tasks)==0:
            break
        def add_step_dim(input):
            if isinstance(input, torch.Tensor):
                return input.unsqueeze(0)
            elif isinstance(input, Dict):
                return {k: add_step_dim(v) for k, v in input.items()}
            else:
                raise NotImplementedError

        batch = add_step_dim(sensor_preprocessor_graph.get_observations(batch_observations([task.get_observations()])))
        #print(batch)
        rnn_outputs = []
        ac_probs = []
        ac_vals = [] 
        rewards = []

        random_rnn_outputs = []
        random_ac_probs = []
        random_ac_vals = [] 
        episode_len = 0
        while not task.is_done():
            episode_len+=1
            random_batch = copy.deepcopy(batch)
            ac_out, memory = cast(
                Tuple[ActorCriticOutput, Memory],
                walkthrough_model.forward(
                    observations=batch,
                    memory=memory,
                    prev_actions=None,
                    masks=masks,
                ),
            )

            random_ac_out, random_memory = cast(
                Tuple[ActorCriticOutput, Memory],
                random_model.forward(
                    observations=random_batch,
                    memory=random_memory,
                    prev_actions=None,
                    masks=random_masks,
                ),
            )
            #print(memory)
            masks = masks.fill_(1.0)
            random_masks = random_masks.fill_(1.0)
            outputs = task.step(
                action=ac_out.distributions.sample().item()
            )

            random_rnn_outputs.append(random_memory['rnn'][0].detach().cpu().numpy().squeeze().tolist())
            random_ac_probs.append(random_ac_out.distributions.probs.detach().cpu().numpy().squeeze().tolist())
            random_ac_vals.append(random_ac_out.values.detach().cpu().numpy().squeeze().tolist())


            rnn_outputs.append(memory['rnn'][0].detach().cpu().numpy().squeeze().tolist())
            ac_probs.append(ac_out.distributions.probs.detach().cpu().numpy().squeeze().tolist())
            ac_vals.append(ac_out.values.detach().cpu().numpy().squeeze().tolist())
            #print(outputs.reward)
            #print(outputs.reward)
            #print(outputs.info)
            #print(ac_out.distributions.sample().item())
            #print(ac_out.distributions.probs[0][0])
            #print(ac_out.distributions.logits[0][0][0])
            #print(ac_out.values[0][0][0],outputs.reward)
            #print(task.task_info)
            
            obs = outputs.observation
            batch = add_step_dim(sensor_preprocessor_graph.get_observations(batch_observations([obs])))
        print(task.task_info['id'],task._success,episode_len)
        count+=1
        if task._success:
            success_count+=1

        random_task_info = copy.deepcopy(task.task_info)

        task.task_info['rnn'] = rnn_outputs
        task.task_info['ac_probs'] = ac_probs
        task.task_info['ac_vals'] = ac_vals
        task_metrics.append(task.task_info)

        random_task_info['rnn'] = random_rnn_outputs
        random_task_info['ac_probs'] = random_ac_probs
        random_task_info['ac_vals'] = random_ac_vals
        random_task_metrics.append(random_task_info)

    
    print("Success is ", success_count/count)
    with open('./train_random.json', 'w') as fout:
        json.dump(random_task_metrics, fout)
    
    with open('./train_' + ckpt_string + '.json', 'w') as fout:
        json.dump(task_metrics, fout)


def pretrained_objectnav_metadata_extraction():
    open_x_displays = []
    try:
        open_x_displays = get_open_x_displays()
    except (AssertionError, IOError):
        pass


    # Define Task sampler 
    objectnav_task_sampler = (
        ObjectNavThorPPOExperimentConfig.make_sampler_fn(
            loop_dataset = False,
            mode="valid",
            seed=12,
            x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            scenes=ObjectNavThorPPOExperimentConfig.VALID_SCENES,
            scene_directory = os.path.join(os.getcwd(), "datasets/ithor-objectnav/val/episodes"),
            object_types= ObjectNavThorPPOExperimentConfig.OBJECT_TYPES,
            env_args= ObjectNavThorPPOExperimentConfig.ENV_ARGS,
            max_steps= ObjectNavThorPPOExperimentConfig.MAX_STEPS,
            sensors= ObjectNavThorPPOExperimentConfig.SENSORS,
            action_space= gym.spaces.Discrete(
                len(ObjectNaviThorGridTask.class_action_names())
            ),
            rewards_config = ObjectNavThorPPOExperimentConfig.REWARD_CONFIG,
        )
    )

    # Define model and load state dict
    tmpdir = "/home/ubuntu/projects/allenact/storage_default/objectnav_ithor_baseline_rs30/checkpoints/ObjectNaviThorPPOResnetGRU/2021-09-01_00-52-40"
    ckpt_path = os.path.join(
            tmpdir, "exp_ObjectNaviThorPPOResnetGRU__stage_00__steps_000040020070.pt"
        )
 
    state_dict = torch.load(
        ckpt_path,
        map_location="cpu",
    )
    mode='valid'
    nprocesses = 1
    sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(ObjectNavThorPPOExperimentConfig.SENSORS).observation_spaces,
                preprocessors=ObjectNavThorPPOExperimentConfig.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )
    walkthrough_model = ObjectNavThorPPOExperimentConfig.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph)
    walkthrough_model.load_state_dict(state_dict["model_state_dict"])
    print("Model Loaded")

    # 

    rollout_storage = RolloutStorage(
            num_steps=1,
            num_samplers=1,
            actor_critic=walkthrough_model,
            only_store_first_and_last_in_memory=False,
        )
    memory = rollout_storage.pick_memory_step(0)
    masks = rollout_storage.masks[:1]

    binned_map_losses = []
    semantic_map_losses = []
    num_tasks = objectnav_task_sampler.length
    count = 0
    success_count =0 
    task_metrics = []
    for i in tqdm(range(5)):
        masks = 0 * masks
        task = objectnav_task_sampler.next_task()
        print((objectnav_task_sampler.max_tasks))
        if (objectnav_task_sampler.max_tasks)==0:
            break
        def add_step_dim(input):
            if isinstance(input, torch.Tensor):
                return input.unsqueeze(0)
            elif isinstance(input, Dict):
                return {k: add_step_dim(v) for k, v in input.items()}
            else:
                raise NotImplementedError

        batch = add_step_dim(sensor_preprocessor_graph.get_observations(batch_observations([task.get_observations()])))
        #print(batch)
        rnn_outputs = []
        ac_probs = []
        ac_vals = [] 
        rewards = []
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
            #print(memory)
            masks = masks.fill_(1.0)
            outputs = task.step(
                action=ac_out.distributions.sample().item()
            )

            rnn_outputs.append(memory['rnn'][0].detach().cpu().numpy().squeeze().tolist())
            ac_probs.append(ac_out.distributions.probs.detach().cpu().numpy().squeeze().tolist())
            ac_vals.append(ac_out.values.detach().cpu().numpy().squeeze().tolist())
            #print(outputs.reward)
            #print(outputs.info)
            #print(ac_out.distributions.sample().item())
            #print(ac_out.distributions.probs[0][0])
            #print(ac_out.distributions.logits[0][0][0])
            #print(ac_out.values[0][0][0],outputs.reward)
            #print(task.task_info)
            
            obs = outputs.observation
            batch = add_step_dim(sensor_preprocessor_graph.get_observations(batch_observations([obs])))
        #print(task.task_info['id'],task._success)
        count+=1
        if task._success:
            success_count+=1
        task.task_info['rnn'] = rnn_outputs
        task.task_info['ac_probs'] = ac_probs
        task.task_info['ac_vals'] = ac_vals
        task_metrics.append(task.task_info)
    
    print("Success is ", success_count/count)
    with open('temp.json', 'w') as fout:
        json.dump(task_metrics, fout)

    #read_metric_file('temp.json')

def test_pretrained_rearrange_walkthrough_mapping_agent( tmpdir):
    try:
        os.chdir(ABS_PATH_OF_TOP_LEVEL_DIR)
        sys.path.append(
            os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "projects/ithor_rearrangement")
        )

        from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
        from baseline_configs.walkthrough.walkthrough_rgb_mapping_ppo import (
            WalkthroughRGBMappingPPOExperimentConfig,
        )
        from rearrange.constants import (
            FOV,
            PICKUPABLE_OBJECTS,
            OPENABLE_OBJECTS,
        )
        from datagen.datagen_utils import get_scenes

        open_x_displays = []
        try:
            open_x_displays = get_open_x_displays()
        except (AssertionError, IOError):
            pass
        walkthrough_task_sampler = (
            WalkthroughRGBMappingPPOExperimentConfig.make_sampler_fn(
                stage="train",
                scene_to_allowed_rearrange_inds={
                    s: [0] for s in get_scenes("train")
                },
                force_cache_reset=True,
                allowed_scenes=None,
                seed=2,
                x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            )
        )

        named_losses = (
            WalkthroughRGBMappingPPOExperimentConfig.training_pipeline().named_losses
        )

        ckpt_path = os.path.join(
            tmpdir, "pretrained_walkthrough_mapping_agent_75mil.pt"
        )
        if not os.path.exists(ckpt_path):
            urllib.request.urlretrieve(
                "https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/walkthrough/pretrained_walkthrough_mapping_agent_75mil.pt",
                ckpt_path,
            )

        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
        )

        walkthrough_model = WalkthroughRGBMappingPPOExperimentConfig.create_model()
        walkthrough_model.load_state_dict(state_dict["model_state_dict"])

        rollout_storage = RolloutStorage(
            num_steps=1,
            num_samplers=1,
            actor_critic=walkthrough_model,
            only_store_first_and_last_in_memory=True,
        )
        memory = rollout_storage.pick_memory_step(0)
        masks = rollout_storage.masks[:1]

        binned_map_losses = []
        semantic_map_losses = []
        for i in range(5):
            masks = 0 * masks

            set_seed(i + 1)
            task = walkthrough_task_sampler.next_task()

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
                print(memory)
                masks = masks.fill_(1.0)
                obs = task.step(
                    action=ac_out.distributions.sample().item()
                ).observation
                batch = add_step_dim(batch_observations([obs]))

                if task.num_steps_taken() >= 10:
                    break
            
        # To save observations for comparison against future runs, uncomment the below.
        # os.makedirs("tmp_out", exist_ok=True)
        # compress_pickle.dump(
        #     {**observations_dict}, "tmp_out/rearrange_mapping_examples.pkl.gz"
        # )
    finally:
        try:
            walkthrough_task_sampler.close()
        except NameError:
            pass


if __name__ == "__main__":
    #test_pretrained_rearrange_walkthrough_mapping_agent("tmp_out")  # Used for local debugging
    random_objectnav_metadata_extraction()
