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

def get_objectnav_ithor_default_resnet(episode_type,ckpt,pretrained=False):

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
            ckpt,
            map_location="cpu",
        )
        walkthrough_model.load_state_dict(state_dict["model_state_dict"])

    return walkthrough_model,objectnav_task_sampler,sensor_preprocessor_graph

def get_model_details(model_id,episode_type,ckpt):
    if model_id == "objectnav_ithor_default_resnet_pretrained":
        model,task_sampler,preprocessor_graph = get_objectnav_ithor_default_resnet(episode_type, ckpt, pretrained = True) 


    return model,task_sampler,preprocessor_graph

def walk_along_trajectory(follower_model_ids, save_dir,episode_type,trajectories_file_path,ckpt,save_metadata=False):

    trajectories_data =  read_metric_file(trajectories_file_path)
    action_list = ['MoveAhead', "RotateLeft", 'RotateRight', "LookDown", "LookUp", 'Stop']
    follower_models, follower_task_samplers, follower_preprocessor_graphs,follower_save_dirs = {},{},{},{}

    if save_metadata:
        follower_metadata = {}
        follower_rnn = {}
        all_obj_types = get_all_object_types()
        action_triplets = get_action_triplets(action_list)
        rotate_step_degrees=30
        num_rotation_angles = math.ceil(360.0 / rotate_step_degrees)
        reachability_radii = [2, 4, 6]  # list(range(1,7))
        grid_size = 0.25


    for follower_model_id in follower_model_ids:
        follower_models[follower_model_id],\
        follower_task_samplers[follower_model_id],\
        follower_preprocessor_graphs[follower_model_id] = get_model_details(follower_model_id,episode_type,ckpt)
        follower_save_dirs[follower_model_id] = os.path.join(save_dir,follower_model_id)
        if not os.path.exists(follower_save_dirs[follower_model_id]):
            os.makedirs(follower_save_dirs[follower_model_id])

        if save_metadata:
            follower_metadata[follower_model_id] = pd.DataFrame()
            follower_rnn[follower_model_id] = pd.DataFrame()


    all_models = copy.deepcopy(follower_models) 

    rollout_storage = {}
    memory = {}
    masks = {}
    task_metrics = {}
    for model_id,model in all_models.items():
        rollout_storage[model_id] = RolloutStorage(
                num_steps=1,
                num_samplers=1,
                actor_critic=model,
                only_store_first_and_last_in_memory=False,
            )
        memory[model_id] = rollout_storage[model_id].pick_memory_step(0)
        masks[model_id] = rollout_storage[model_id].masks[:1]
        task_metrics[model_id] = []

    episode_list = list(trajectories_data.keys())
    num_tasks = len(episode_list)
    
    count = 0
    success_count =0 
    
    for i in tqdm(range(num_tasks)):

        for model_id,model in all_models.items():
            masks[model_id]  = 0 * masks[model_id]
                
        follower_tasks = {}
        for f,follower_model_id in enumerate(follower_model_ids):
            scene = episode_list[i].split("_")[0]
            episode = episode_list[i]
            print(scene,episode)
            follower_tasks[follower_model_id] = follower_task_samplers[follower_model_id].next_task_from_info(scene,episode)
            init_event = follower_tasks[follower_model_id].env.controller.last_event
            follower_tasks[follower_model_id].env.controller.step("PausePhysicsAutoSim")
            

        if follower_tasks[follower_model_id] is None:
            print(follower_tasks[follower_model_id])
            break

        print((follower_task_samplers[follower_model_id].max_tasks))
        if (follower_task_samplers[follower_model_id].max_tasks)==0:
            break

        follower_batches = {}
        for follower_model_id in follower_model_ids:
            batch = \
            follower_preprocessor_graphs[follower_model_id].get_observations(batch_observations([follower_tasks[follower_model_id].get_observations()]))
            follower_batches[follower_model_id] = add_step_dim(batch)
        
        all_batches = copy.deepcopy(follower_batches) 

        rnn_outputs,ac_probs,ac_vals = {},{},{}

        for model_id,model in all_models.items():
            rnn_outputs[model_id] = []
            ac_probs[model_id] = []
            ac_vals[model_id] = []


        episode_len = 0
        first_2_actions = []
        traj_feats = {}
        ep_details = []
        #plt.figure()
        while not follower_tasks[follower_model_id].is_done():
            
            ac_out = {}
            for model_id,model in all_models.items():
                with torch.no_grad():
                    ac_out[model_id], memory[model_id] = cast(
                        Tuple[ActorCriticOutput, Memory],
                        model.forward(
                            observations=all_batches[model_id],
                            memory=memory[model_id],
                            prev_actions=None,
                            masks=masks[model_id],
                        ),
                    )

                masks[model_id] = masks[model_id].fill_(1.0)
                rnn_outputs[model_id].append(memory[model_id]['rnn'][0].detach().cpu().numpy().squeeze().tolist())
                ac_probs[model_id].append(ac_out[model_id].distributions.probs.detach().cpu().numpy().squeeze().tolist())
                ac_vals[model_id].append(ac_out[model_id].values.detach().cpu().numpy().squeeze().tolist())
            
            


            #plt.imshow(follower_tasks[first_model_id].env.controller.last_event.frame)
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            #print("Task is ", follower_tasks[model_id].task_info['id'])

            next_action = action_list.index(trajectories_data[follower_tasks[model_id].task_info['id']]['actions_taken'][episode_len]['action'])
            
            for follower_model_id in follower_model_ids:
                outputs = follower_tasks[follower_model_id].step(
                    action = next_action
                )
                if "pointnav" in follower_model_id:
                    print("Distance to pointnav target", follower_tasks[follower_model_id].dist_to_target())
                obs = outputs.observation
                batch = \
                follower_preprocessor_graphs[follower_model_id].get_observations(batch_observations([obs]))
                follower_batches[follower_model_id] = add_step_dim(batch)

            all_batches = copy.deepcopy(follower_batches)        
            #print("Did collision happen? ", not follower_tasks[follower_model_id].env.controller.last_event.metadata["lastActionSuccess"])
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


            if save_metadata:
                if episode_len<2:
                    first_2_actions.append(action_list[next_action])
                elif episode_len == 2:
                    for model_id,model in all_models.items():
                        if 'pointnav' in model_id:
                            traj_feats[model_id] = trajectory_metadata_pointnav(init_event,
                                                            first_2_actions,
                                                            all_obj_types,
                                                            action_triplets,
                                                            reachability_radii,
                                                            num_rotation_angles,
                                                            rotate_step_degrees,
                                                            grid_size,
                                                            follower_tasks[model_id].task_info,
                                                            follower_tasks[model_id].env.controller,
                                                            )
                        else: 
                            traj_feats[model_id] = trajectory_metadata(init_event,
                                                            first_2_actions,
                                                            all_obj_types,
                                                            action_triplets,
                                                            reachability_radii,
                                                            num_rotation_angles,
                                                            rotate_step_degrees,
                                                            grid_size,
                                                            follower_tasks[model_id].task_info,
                                                            follower_tasks[model_id].env.controller,
                                                            )
                if episode_len >= 2: # and not i == len(agent_actions_taken) - 1:
                    # get trajectory metadata
                    for model_id,model in all_models.items():
                        collision_info = not follower_tasks[model_id].env.controller.last_event.metadata["lastActionSuccess"]
                        
                        traj_feats[model_id].update_dicts(  follower_tasks[model_id].env.controller.last_event,
                                                            action_list[next_action],                                                            
                                                            collision_info)
                    step_info = follower_tasks[model_id].task_info['id'] + "_actionstep_" + str(episode_len).zfill(3)
                    ep_details.append(step_info)

            episode_len+=1
        print(follower_tasks[model_id].task_info['id'],follower_tasks[model_id]._success,episode_len)
        count+=1
        if follower_tasks[model_id]._success:
            success_count+=1
        task_info ={}

        

        for model_id,model in all_models.items():
            task_info[model_id] = copy.deepcopy(follower_tasks[model_id].task_info)
            task_info[model_id]['rnn'] = rnn_outputs[model_id]
            task_info[model_id]['ac_probs'] = ac_probs[model_id]
            task_info[model_id]['ac_vals'] = ac_vals[model_id]
            task_metrics[model_id].append(task_info[model_id])
            save_path = os.path.join(follower_save_dirs[model_id], model_id + '_f.json' )
            with open(save_path, 'w') as fout:
                json.dump(task_metrics[model_id], fout)

            if save_metadata:
                if episode_len > 2:
                    episode_df = pd.DataFrame(traj_feats[model_id].trajectory_feat_list, index=ep_details)
                    rnn_df = pd.DataFrame(rnn_outputs[model_id][2:], index=ep_details)
                    ac_prob_labels = ['AC_MOVE_AHEAD', 'AC_ROTATE_LEFT', 'AC_ROTATE_RIGHT', 'AC_LOOK_DOWN', 'AC_LOOK_UP', 'AC_END'] 
                    ac_policy_labels = ['AC_POLICY']
                    ac_prob_df = pd.DataFrame(ac_probs[model_id][2:], index=ep_details, columns = ac_prob_labels)
                    ac_policy_df = pd.DataFrame(ac_vals[model_id][2:], index=ep_details, columns = ac_policy_labels)
                    episode_df = pd.concat([episode_df, ac_prob_df,ac_policy_df], axis=1)

                follower_metadata[model_id] = follower_metadata[model_id].append(episode_df)
                follower_rnn[model_id] = follower_rnn[model_id].append(rnn_df)
                follower_metadata[model_id].to_pickle(os.path.join(follower_save_dirs[model_id],"metadata.pkl"))
                follower_rnn[model_id].to_pickle(os.path.join(follower_save_dirs[model_id],"rnn.pkl"))

        #for model_id,model in all_models.items():
        #    assert are_trajectories_same(task_info[model_id]["followed_path"],task_info[first_model_id]["followed_path"])




    print("Success is ", success_count/count)

    for model_id,model in all_models.items():
        follower_tasks[model_id].env.controller.stop()
    #    save_path = os.path.join(follower_save_dirs[model_id],'all_episodes_f.json' )
    #    with open(save_path, 'w') as fout:
    #        json.dump(task_metrics[model_id], fout)

    #    if save_metadata:
    #        follower_metadata[model_id].to_pickle(os.path.join(follower_save_dirs[model_id],"all_metadata.pkl"))
    #        follower_rnn[model_id].to_pickle(os.path.join(follower_save_dirs[model_id],"all_rnn.pkl"))



def walk_along_random_actions():
    return 0


open_x_displays = []
try:
    open_x_displays = get_open_x_displays()
except (AssertionError, IOError):
    pass

def main():

    parser = argparse.ArgumentParser(description='Generates model activations following a model/human trajectory')
    parser.add_argument('-e','--episode_type', help='run on which episodes val/train', default = 'val', type=str)
    parser.add_argument('-m','--arch',help='architecture : resnet or simple conv', default = 'resnet', type=str)
    parser.add_argument('--active', help = 'Which model is active pointnav/objectnav/human',default = 'objectnav', type=str )
    parser.add_argument('-tf','--trajectories_file', help = 'path to trajectory file for following',default = '', type=str )

    args = vars(parser.parse_args())

    model_types = ['objectnav','pointnav']

    follower_model_ids = ["objectnav_ithor_default_" + args['arch'] + "_pretrained" ]
    save_dir = os.path.join('trajectory_metadata_all+ckpts',args['episode_type'],'active_human_sub1_trajectories_with_metadata')
    ckpt_dir = "/home/ubuntu/projects/allenact/storage_default/objectnav_ithor_baseline_rs30/checkpoints/ObjectNaviThorPPOResnetGRU/2021-09-01_00-52-40"
    all_ckpts = glob.glob(ckpt_dir +'/*.pt')
    print(len(all_ckpts))
    all_ckpts.sort()
    for ckpt in tqdm(all_ckpts[:54]):
        ckpt_id = ckpt.split("/")[-1].split(".")[0]
        print(ckpt_id)
        save_dir = os.path.join('trajectory_metadata_all_ckpts',args['episode_type'],'active_human_sub1_trajectories_with_metadata',ckpt_id)
        #print(save_dir)
        walk_along_trajectory(follower_model_ids, save_dir,args['episode_type'],args['trajectories_file'], ckpt, save_metadata=True)
    


if __name__ == "__main__":
    main()