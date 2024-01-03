import numpy as np
import os

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
#from ray.tune.logger import pretty_print

from rl.environment_char import Environment
from rl.MyCallBack import MyCallBack

# from rl.environment_velocity import Environment

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

            'normalize_actions': False,
            # 'callbacks': DefaultCallbacks, # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
            'callbacks':MyCallBack,

            'ignore_worker_failures': False,

            'vf_clip_param':5000,

            'horizon':1000,

            'rollout_fragment_length':256,
            'train_batch_size':4096,
            'sgd_minibatch_size':256,

        }
        config.update(custom_config)

#        config['model']['fcnet_hiddens'] = [16, 16]
        config['model']['fcnet_hiddens'] = [32, 32]
#        config['model']['fcnet_hiddens'] = [64, 64]
#        config['model']['fcnet_hiddens'] = [128, 128]
#        config['model']['fcnet_hiddens'] = [256, 256]

        def env_creater(env_config):
            return Environment({})
        register_env("my_env_char",env_creater)

        self.trainer = ppo.PPOTrainer(config=config, env="my_env_char")

        if path: #restore
            self.trainer.restore(path)


    def learn(self, iterations):
        CHECKPOINT = 50
        try:
            while True:
                result = self.trainer.train()

#                print(pretty_print(result))
                print("training_iteration: %d"%result["training_iteration"])
                print("date: ", result["date"])
                print("timesteps_total: %f"%result["timesteps_total"])
                print("time_total_s: %f"%result["time_total_s"])
                print()
                print("episode_len_mean: %f"%result["episode_len_mean"])
                print("episode_reward_min: %f"%result["episode_reward_min"])
                print("episode_reward_mean: %f"%result["episode_reward_mean"])
                print("episode_reward_max: %f"%result["episode_reward_max"])
                print("\n\n")

                i = result['training_iteration']
                if i % CHECKPOINT == 0:
                    print('Checkpoint saved at', self.trainer.save())
                if i >= iterations:
                #if result['time_total_s'] >= 600:
                    break
            print('Last model saved at', self.trainer.save())
        except KeyboardInterrupt:
            print('Checkpoint saved at', self.trainer.save())
        ray.shutdown()

    def action(self, obs):
        a = self.trainer.compute_single_action(observation=obs, explore=False) # policy_id="default_policy"
        assert(len(a) == 12)
        return a

