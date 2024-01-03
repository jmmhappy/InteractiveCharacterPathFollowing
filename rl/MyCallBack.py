from typing import Dict, Tuple
import argparse
import numpy as np
import os

import ray
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker

from ray.rllib.agents.callbacks import DefaultCallbacks
#from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class MyCallBack(DefaultCallbacks):

    def on_learn_on_batch(self, *, policy:Policy, train_batch:SampleBatch, result:dict, **kwargs) ->None:
        taskRewards = [d['taskReward'] for d in train_batch['infos']] 
        result["mean_task_rewards_in_train_batch"] = np.mean(taskRewards)

        distances = np.array([d['initialDistance'] for d in train_batch['infos']])
        steps = train_batch.agent_steps() # batch size
        stepsPerMeter = np.mean(distances/steps)
        result["steps_per_meter_in_train_batch"] = stepsPerMeter

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        print(base_env.cur_infos)
#        episode.custom_metrics["steps_per_meter_in_episode"] = /episode.length
