from __future__ import print_function  # for Python2

import os
import platform
import random
import sys
import urllib
import urllib.request
import warnings
import copy
import readchar
import argparse
import torch
from collections import defaultdict
from math import radians

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
import matplotlib.pyplot as plt

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

from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from constants import ABS_PATH_OF_TOP_LEVEL_DIR

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
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask,PointNaviThorTask
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
import sys

from trajectory_features.trajectory_metadata import trajectory_metadata,trajectory_metadata_pointnav
from trajectory_features.extract_trajectory_metadata import get_all_object_types,get_action_triplets
from trajectory_features.thor_utils import *
from trajectory_features.visualization_utils import *
from trajectory_features import __version__
from trajectory_features.headless_controller import HeadlessController

import json
import pandas as pd

allenact_to_ai2thor_actions = {
    "MoveAhead" : "MoveAhead",
    "RotateRight" : "RotateRight",
    "RotateLeft" : "RotateLeft",
    "LookUp" : "LookUp",
    "LookDown" : "LookDown",
    "End" : "Stop"
} 

from allenact_plugins.ithor_plugin.ithor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
)

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
    return challenge_metrics["episodes"]

def are_trajectories_same(path1,path2):
    print("are paths equal", path1==path2)
    return path1==path2

def add_step_dim(input):
    if isinstance(input, torch.Tensor):
        return input.unsqueeze(0)
    elif isinstance(input, Dict):
        return {k: add_step_dim(v) for k, v in input.items()}
    else:
        raise NotImplementedError

def get_collected_episodes(metric_file):
    with open(metric_file, "r") as read_file:
        allenact_metrics = json.load(read_file)
    collected_episodes = []
    for episode in allenact_metrics:
        collected_episodes.append([episode["id"]])
    return collected_episodes
    
def get_action_from_human():
    key2action = {
        "'q'": "MoveAhead", 
        "'a'": "RotateLeft" ,
        "'d'": "RotateRight" , 
        "'s'": "LookDown" , 
        "'w'": "LookUp" , 
        "'e'": "Stop" , 
    }
    possible_actions = ["MoveAhead", "RotateLeft", "RotateRight", "LookDown", "LookUp", "Stop"]
    key = repr(readchar.readchar())
    while key not in key2action.keys():
        key = repr(readchar.readchar())
        print("Press one of the action keys")
        print(key2action)

    action = possible_actions.index(key2action[key])
    print("next action is ", action, key2action[key])
    return  action

def get_objectnav_ithor_default_resnet(episode_type,pretrained=False):

    from projects.tutorials.object_nav_ithor_ppo_baseline import ObjectNavThorPPOExperimentConfig
    # Define Task sampler 

    if episode_type == 'train':
        episode_type_string = 'train_sampled'
    else:
        episode_type_string = episode_type


    val_path = os.path.join(
        "datasets/ithor-objectnav/" + episode_type_string + "/episodes", "*.json.gz"
    )    
    scenes = [
        os.path.basename(scene).split(".")[0] for scene in glob.glob(val_path)
    ]
    objectnav_task_sampler = (
        ObjectNavThorPPOExperimentConfig.make_sampler_fn(
            loop_dataset = False,
            mode="valid",
            seed=12,
            x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            scenes=scenes,
            scene_directory = os.path.join(os.getcwd(),"datasets/ithor-objectnav",episode_type_string,"episodes"),
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
    if pretrained:
        # Define model and load state dict
        tmpdir = "./pretrained_model_ckpts/"
        ckpt_string = "exp_ObjectNaviThorPPOResnetGRU__stage_00__steps_000050025710"
        ckpt_path = os.path.join(
                tmpdir, ckpt_string + ".pt"
            )
    
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
        )
        walkthrough_model.load_state_dict(state_dict["model_state_dict"])

    return walkthrough_model,objectnav_task_sampler,sensor_preprocessor_graph
    
def get_objectnav_ithor_default_simpleconv(episode_type,pretrained=False):
    
    from projects.tutorials.object_nav_ithor_ppo_simpleconv import ObjectNavThorPPOExperimentConfig
    # Define Task sampler 
    if episode_type == 'train':
        episode_type_string = 'train_sampled'
    else:
        episode_type_string = episode_type
    val_path = os.path.join(
        "datasets/ithor-objectnav/" + episode_type_string + "/episodes", "*.json.gz"
    )    
    scenes = [
        os.path.basename(scene).split(".")[0] for scene in glob.glob(val_path)
    ]

    objectnav_task_sampler = (
        ObjectNavThorPPOExperimentConfig.make_sampler_fn(
            loop_dataset = False,
            mode="valid",
            seed=12,
            x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            scenes=scenes,
            scene_directory = os.path.join(os.getcwd(), "datasets/ithor-objectnav",episode_type_string,"episodes"),
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
    if pretrained:
        # Define model and load state dict
        tmpdir = "./pretrained_model_ckpts/"
        ckpt_string = "exp_ObjectNaviThorPPOSimpleConvGRU__stage_00__steps_000050030700"
        ckpt_path = os.path.join(
                tmpdir, ckpt_string + ".pt"
            )
    
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
        )
        walkthrough_model.load_state_dict(state_dict["model_state_dict"])

    return walkthrough_model,objectnav_task_sampler,sensor_preprocessor_graph
    
def get_pointnav_ithor_default_resnet(episode_type,pretrained=False):

    from projects.tutorials.point_nav_ithor_ppo_baseline import PointNavThorPPOExperimentConfig

    val_path = os.path.join(
        "datasets/ithor-pointnav-on-objectnav/" + episode_type + "/episodes", "*.json.gz"
    )
    scenes = [
        os.path.basename(scene).split(".")[0] for scene in glob.glob(val_path)
    ]
    # Define Task sampler 
    task_sampler = (
        PointNavThorPPOExperimentConfig.make_sampler_fn(
            loop_dataset = False,
            mode="valid",
            seed=12,
            x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            scenes=scenes,
            scene_directory = os.path.join(os.getcwd(), "datasets/ithor-pointnav-on-objectnav/" +episode_type+"/"),
            env_args= PointNavThorPPOExperimentConfig.ENV_ARGS,
            max_steps= PointNavThorPPOExperimentConfig.MAX_STEPS,
            sensors= PointNavThorPPOExperimentConfig.SENSORS,
            action_space= gym.spaces.Discrete(
                len(PointNaviThorTask.class_action_names())
            ),
            rewards_config = PointNavThorPPOExperimentConfig.REWARD_CONFIG,
        )
    )

    
    mode='valid'
    nprocesses = 1
    sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(PointNavThorPPOExperimentConfig.SENSORS).observation_spaces,
                preprocessors=PointNavThorPPOExperimentConfig.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )
    walkthrough_model = PointNavThorPPOExperimentConfig.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph)
    if pretrained:
        # Define model and load state dict
        tmpdir = "./pretrained_model_ckpts/"
        ckpt_string = "exp_PointNaviThorPPOResnetGRU__stage_00__steps_000050017230"
        ckpt_path = os.path.join(
                tmpdir, ckpt_string + ".pt"
            )
    
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
        )
        walkthrough_model.load_state_dict(state_dict["model_state_dict"])

    return walkthrough_model,task_sampler,sensor_preprocessor_graph

def get_pointnav_ithor_default_simpleconv(episode_type, pretrained=False):

    from projects.tutorials.point_nav_ithor_ppo_simpleconv import PointNavThorPPOExperimentConfig

    val_path = os.path.join(
        os.getcwd(), "datasets/ithor-pointnav-on-objectnav/" + episode_type + "/episodes", "*.json.gz"
    )
    scenes = [
        os.path.basename(scene).split(".")[0] for scene in glob.glob(val_path)
    ]
    # Define Task sampler 
    task_sampler = (
        PointNavThorPPOExperimentConfig.make_sampler_fn(
            loop_dataset = False,
            mode="valid",
            seed=12,
            x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            scenes=scenes,
            scene_directory = os.path.join(os.getcwd(), "datasets/ithor-pointnav-on-objectnav/" +episode_type+"/"),
            env_args= PointNavThorPPOExperimentConfig.ENV_ARGS,
            max_steps= PointNavThorPPOExperimentConfig.MAX_STEPS,
            sensors= PointNavThorPPOExperimentConfig.SENSORS,
            action_space= gym.spaces.Discrete(
                len(PointNaviThorTask.class_action_names())
            ),
            rewards_config = PointNavThorPPOExperimentConfig.REWARD_CONFIG,
        )
    )

    
    mode='valid'
    nprocesses = 1
    sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(PointNavThorPPOExperimentConfig.SENSORS).observation_spaces,
                preprocessors=PointNavThorPPOExperimentConfig.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )
    walkthrough_model = PointNavThorPPOExperimentConfig.create_model(sensor_preprocessor_graph=sensor_preprocessor_graph)
    if pretrained:
        # Define model and load state dict
        tmpdir = "./pretrained_model_ckpts/"
        ckpt_string = "exp_PointNaviThorPPOSimpleConvGRU__stage_00__steps_000050018780"
        ckpt_path = os.path.join(
                tmpdir, ckpt_string + ".pt"
            )
    
        state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
        )
        walkthrough_model.load_state_dict(state_dict["model_state_dict"])

    return walkthrough_model,task_sampler,sensor_preprocessor_graph

def get_model_details(model_id,episode_type):
    if model_id == "pointnav_ithor_default_resnet_pretrained":
        model,task_sampler,preprocessor_graph = get_pointnav_ithor_default_resnet(episode_type, pretrained = True) 
    elif model_id == "pointnav_ithor_default_simpleconv_pretrained":
        model,task_sampler,preprocessor_graph = get_pointnav_ithor_default_simpleconv(episode_type, pretrained = True) 
    elif model_id == "pointnav_ithor_default_resnet_random":
        model,task_sampler,preprocessor_graph = get_pointnav_ithor_default_resnet(episode_type,) 
    elif model_id == "pointnav_ithor_default_simpleconv_random":
        model,task_sampler,preprocessor_graph = get_pointnav_ithor_default_simpleconv(episode_type,) 
    elif model_id == "objectnav_ithor_default_resnet_pretrained":
        model,task_sampler,preprocessor_graph = get_objectnav_ithor_default_resnet(episode_type, pretrained = True) 
    elif model_id == "objectnav_ithor_default_simpleconv_pretrained":
        model,task_sampler,preprocessor_graph = get_objectnav_ithor_default_simpleconv(episode_type, pretrained = True) 
    elif model_id == "objectnav_ithor_default_resnet_random":
        model,task_sampler,preprocessor_graph = get_objectnav_ithor_default_resnet(episode_type,) 
    elif model_id == "objectnav_ithor_default_simpleconv_random":         
        model,task_sampler,preprocessor_graph = get_objectnav_ithor_default_simpleconv(episode_type,) 

    return model,task_sampler,preprocessor_graph

def visualize_hidden_unit(model_id, save_dir,episode_type,hidden_unit_ids):

    action_list = ['MoveAhead', "RotateLeft", 'RotateRight', "LookDown", "LookUp", 'Stop']

    all_obj_types = get_all_object_types()
    action_triplets = get_action_triplets(action_list)
    rotate_step_degrees=30
    num_rotation_angles = math.ceil(360.0 / rotate_step_degrees)
    reachability_radii = [2, 4, 6]  # list(range(1,7))
    grid_size = 0.25

    model, task_sampler, preprocessor_graph = get_model_details(model_id,episode_type)

    rollout_storage = RolloutStorage(
                num_steps=1,
                num_samplers=1,
                actor_critic=model,
                only_store_first_and_last_in_memory=False,
            )
    memory = rollout_storage.pick_memory_step(0)
    masks = rollout_storage.masks[:1]

    num_tasks = 5
    
    for i in tqdm(range(num_tasks)):


        masks  = 0 * masks      

        task = task_sampler.next_task()
        init_event = task.env.controller.last_event
        episode = task.task_info['id']
                
        batch = preprocessor_graph.get_observations(batch_observations([task.get_observations()]))
        batch = add_step_dim(batch)
        
        episode_len = 0
        

        first_2_actions = []
        fig, axs = plt.subplots(
                nrows=2, ncols=3, figsize=(3 * 3, 3 * 2),gridspec_kw={'width_ratios': [1,3,4.4]}
                )

        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        for i in range(3):
            axs[1][i].remove()
        while not task.is_done():
            with torch.no_grad():
                ac_out, memory = cast(
                    Tuple[ActorCriticOutput, Memory],
                    model.forward(
                        observations=batch,
                        memory=memory,
                        prev_actions=None,
                        masks=masks,
                    ),
                )

                masks = masks.fill_(1.0)

            #plt.imshow(follower_tasks[first_model_id].env.controller.last_event.frame)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Task is ", task.task_info['id'])
            next_action = get_action_from_human()
            
            
            outputs = task.step(action = next_action)
            if "pointnav" in model_id:
                print("Distance to pointnav target", task.dist_to_target())
            obs = outputs.observation
            batch = preprocessor_graph.get_observations(batch_observations([obs]))
            batch = add_step_dim(batch)
            print("Did collision happen? ", not task.env.controller.last_event.metadata["lastActionSuccess"])
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


            if episode_len<2:
                first_2_actions.append(action_list[next_action])
            elif episode_len == 2:
                if 'pointnav' in model_id:
                    traj_feats = trajectory_metadata_pointnav(init_event,
                                                    first_2_actions,
                                                    all_obj_types,
                                                    action_triplets,
                                                    reachability_radii,
                                                    num_rotation_angles,
                                                    rotate_step_degrees,
                                                    grid_size,
                                                    task.task_info,
                                                    task.env.controller,
                                                    )
                else: 
                    traj_feats = trajectory_metadata(init_event,
                                                    first_2_actions,
                                                    all_obj_types,
                                                    action_triplets,
                                                    reachability_radii,
                                                    num_rotation_angles,
                                                    rotate_step_degrees,
                                                    grid_size,
                                                    task.task_info,
                                                    task.env.controller,
                                                    )
            if episode_len >= 2: # and not i == len(agent_actions_taken) - 1:
                
                collision_info = not task.env.controller.last_event.metadata["lastActionSuccess"]
                
                traj_feats.update_dicts(task.env.controller.last_event,
                                        action_list[next_action],                                                            
                                        collision_info)

                metadata2plot = get_metadata2plot(traj_feats)
                frame = task.env.controller.last_event.frame
                map_data = get_agent_map_data(task.env.controller)

                trajectory = [{
                                    "x" : p["x"],
                                    "y" : p["y"],
                                    "z" : p["z"],
                                    "rotation" : p["rotation"]["y"],
                                    "horizon" : p["horizon"]
                                } for p in task.task_info['followed_path']]

                topdown_frame = visualize_agent_path(trajectory,
                                                    task.task_info['target'],
                                                    map_data['frame'],
                                                    map_data['pos_translator'],
                                                    color_pair_ind=episode_len,
                                                    )
                rnn = memory['rnn'][0].detach().cpu().numpy().squeeze().tolist()

                for i in range(1):
                    for j in range(3):
                        axs[i][j].clear()
                plot_visualization(axs,fig,task.task_info['id'], frame, rnn,topdown_frame,metadata2plot, hidden_unit_ids)
                plt.pause(0.05)
                plt.draw() 
                

            episode_len+=1

def get_metadata2plot(traj_feats):
    #target and agent r,theta update
    metadata = {}
    metadata['r'] = [data['r'] for data in traj_feats.trajectory_feat_list]
    metadata['theta'] = [data['theta'] for data in traj_feats.trajectory_feat_list]
    metadata['target_r'] =[data['target_r'] for data in traj_feats.trajectory_feat_list]
    metadata['target_theta'] = [data['target_theta'] for data in traj_feats.trajectory_feat_list]


    #reachability update
    rotate_step_degrees = 30
    num_rotation_angles = math.ceil(360.0 / rotate_step_degrees)
    reachability_radii = [2, 4, 6]  # list(range(1,7))
    metadata['reachability_r']=[]
    metadata['reachability_theta']=[]
    metadata['obstacles_r']=[]
    metadata['obstacles_theta']=[]


    for radius in reachability_radii:
        for rotation_angle in range(num_rotation_angles):
            dict_key = "reachable_R=" + str(radius) + "_theta=" + \
                    str(int(rotation_angle * rotate_step_degrees)).zfill(3)
            if radius<4:
                print(dict_key,traj_feats.trajectory_feat_list[-1][dict_key])
            
            if traj_feats.trajectory_feat_list[-1][dict_key]:
                metadata['reachability_r'].append(radius)
                metadata['reachability_theta'].append(rotation_angle*30)
            else:
                metadata['obstacles_r'].append(radius)
                metadata['obstacles_theta'].append(rotation_angle*30) 
    return metadata




def plot_visualization(axs,fig,episode_id,frame, rnn, topdown_frame, metadata, hidden_unit_ids):
    ax = axs[0][0]
    rnn_value_to_show = []
    units_to_show = []
    for unit in hidden_unit_ids:
        units_to_show.append(str(unit))
        rnn_value_to_show.append(rnn[unit])
    ax.bar(units_to_show,rnn_value_to_show)
    ax.set_xticklabels(units_to_show, rotation=90)
    ax.set_title("RNN output",fontsize=8)
    ax.set_ylim(-1,1)
    ax.grid(False)
    #ax.axhline(y=0, color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    

    ax = axs[0][1]
    im = ax.imshow(frame)
    ax.axis("off")
    ax.set_title("RGB",y=1,fontsize=8)


    ax = axs[0][2]
    im = ax.imshow(topdown_frame)
    ax.axis("off")
    ax.set_title("Top down view",y=1,fontsize=8)
    
    
    
    ax = fig.add_subplot(2, 3, 4, projection='polar')
    ax.clear()
    ax.set_theta_zero_location("N")
    ax.set_ylim(0,4)
    thetas = [radians(a) for a in metadata['theta']]
    Rs = metadata['r']
    ax.plot(thetas,Rs,lw=1.5,color='k')
    ax.scatter(thetas[-1:],Rs[-1:],lw=1.5,color='g')
    ax.set_title("Agents Orientation wrt init",fontsize=8)
    
    
    ax = fig.add_subplot(2, 3, 5, projection='polar')
    ax.clear()
    ax.set_theta_zero_location("N")
    ax.set_ylim(0,4)
    thetas = [radians(a) for a in metadata['target_theta']]
    Rs = metadata['target_r']
    ax.plot(thetas,Rs,lw=1.5,color='b')
    ax.scatter(thetas[-1:],Rs[-1:],lw=1.5,color='g')
    ax.set_title("Agents Orientation wrt target",fontsize=8)
    
    ax = fig.add_subplot(2, 3, 6, projection='polar')
    ax.clear()
    ax.set_theta_zero_location("N")
    #ax.set_theta_direction(-1)
    ax.set_ylim(0,10)
    
    reachable_thetas = [radians(a) for a in metadata['reachability_theta']]
    reachable_Rs = metadata['reachability_r']
    ax.scatter(reachable_thetas,reachable_Rs,lw=1.0,color='g',label='free')

    obstacles_thetas = [radians(a) for a in metadata['obstacles_theta']]
    obstacles_Rs = metadata['obstacles_r']
    ax.scatter(obstacles_thetas,obstacles_Rs,lw=1.0,color='r',label='obstacles')
    ax.set_title("Reachable positions",fontsize=8)
    ax.legend(loc="upper right",prop={'size': 6})
    
    title_string = episode_id
    fig.suptitle(title_string,y = 1,fontsize=10,fontweight='bold')

open_x_displays = []
try:
    open_x_displays = get_open_x_displays()
except (AssertionError, IOError):
    pass

def main():

    parser = argparse.ArgumentParser(description='Generates model activations following a model/human trajectory')
    parser.add_argument('-m','--arch',help='architecture : resnet or simple conv', default = 'simpleconv', type=str)
    parser.add_argument('-t','--task',help='task : objectnav or pointnav', default = 'objectnav', type=str)
    parser.add_argument('-id','--hidden_unit_ids',nargs="*", help='ids of the hidden unit', default = [0], type=int)
    
    args = vars(parser.parse_args())

    model_id = args['task'] + "_ithor_default_" + args['arch'] + "_pretrained"

    save_dir = os.path.join('trajectory_metadata','val','interactive_hidden_unit_visualization')
    visualize_hidden_unit(model_id, save_dir,'val',args['hidden_unit_ids'])

   

if __name__ == "__main__":
    main()