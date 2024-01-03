import numpy as np
import os

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
#from ray.tune.logger import pretty_print

from rl.environment import Environment
from data.Obstacle import ObstacleMap, SIZE

#from networks.MapTorchNetwork import MapTorchNetwork
from networks.MapSmallerNetwork import MapTorchNetwork

class Agent:
    def __init__(self, path=None):

        temp_dir = '/home/jmmhappy' 
        ray.init(_temp_dir=temp_dir, num_cpus=16, log_to_driver=True)
        # ray.init(num_cpus=1, log_to_driver=True)

        if path:
            workers, env_per_workers = 1,1
        else:
            workers, env_per_workers = 16,1

        config = ppo.DEFAULT_CONFIG.copy()
        custom_config = {
            'num_gpus': 1,
            'num_workers': workers,
            'num_envs_per_worker': env_per_workers,
#            'create_env_on_driver': False,
            'framework': 'torch',
            'env_config': {'map':ObstacleMap(), 'map_size':SIZE},

            'normalize_actions': False,
        #    'callbacks': DefaultCallbacks, # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
            'ignore_worker_failures': False,

            'vf_clip_param':5000,

            'horizon':1000,

            'rollout_fragment_length':256,
            'train_batch_size':256*workers,
            'sgd_minibatch_size':256,

            'model':{
                'vf_share_layers':False, # Does nothing(custom model)
                'custom_model': MapTorchNetwork,
                'custom_model_config': {},
            }
        }
        config.update(custom_config)

        def env_creater(env_config):
            return Environment(env_config)
        register_env("my_env_map",env_creater)

        self.trainer = ppo.PPOTrainer(config=config, env="my_env_map")

        if path: #restore
            self.trainer.restore(path)


    def learn(self, iterations):
        CHECKPOINT = 100
        try:
            while True:
                result = self.trainer.train()

                print("training_iteration: %d"%result["training_iteration"])
                print("date: ", result["date"])
                print("episode_len_mean: %f"%result["episode_len_mean"])
                print("episode_reward_min: %f"%result["episode_reward_min"])
                print("episode_reward_mean: %f"%result["episode_reward_mean"])
                print("episode_reward_max: %f"%result["episode_reward_max"])
                print("\n\n")
#                print(pretty_print(result))

                i = result['training_iteration']
                if i % CHECKPOINT == 0:
                    print('Checkpoint saved at', self.trainer.save())
                if i >= iterations:
                    break
            print('Last model saved at', self.trainer.save())
        except KeyboardInterrupt:
            print('Checkpoint saved at', self.trainer.save())
        ray.shutdown()

    def action(self, obs):
        a = self.trainer.compute_single_action(observation=obs, explore=False) # policy_id="default_policy"
        assert(len(a) == 10)
        return a

