
#import pytorch_model_summary
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (SlimConv2d, SlimFC, same_padding, normc_initializer)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

import numpy as np

torch, nn = try_import_torch()

class MapTorchNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, None, model_config, name) 

        #self.num_outputs = num_outputs 

        obs_space = obs_space.original_space

#        in_size = np.array([32,32]) # map
        def _generate_conv_model():
            in_channels = 1 # binary map
            conv = []
            for out_channels, kernel, stride in [[16, [5,5], 1], [32, [4,4], 1], [32, [3,3], 1]]:
#                padding, out_size = same_padding(in_size, kernel, stride)
                conv.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding=None,
                        activation_fn="relu",
                    )
                )
#                in_size = out_size
                in_channels = out_channels

            conv.append(nn.Flatten())
            conv.append(
                SlimFC(
                    in_size=16928, # 23*23*32
                    out_size=128,
                    activation_fn="relu",
                )
            )
            return conv

        def _generate_linear_model():
            # together with char state and goal (17)
            prev_layer_size = 128 + obs_space['character'].shape[0]

            linear = []
            for size in [512, 256]:
                linear.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        activation_fn="relu",
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_layer_size = size
            return linear
     

        conv, linear = _generate_conv_model(), _generate_linear_model()
        self.layer1 = nn.Sequential(*conv).float()
        self.layer2 = nn.Sequential(*linear).float()

        # final layer fc to output(12)
        self._final_layer = SlimFC(
                in_size=256,#prev_layer_size,
                out_size=num_outputs,
                activation_fn=None,
                initializer=normc_initializer(1.0)
        )

#        print(pytorch_model_summary.summary(self.layer1, torch.zeros(32, 1, 32, 32), show_input=True))
#        print(pytorch_model_summary.summary(self.layer1, torch.zeros(32, 1, 32, 32), show_input=False))
#        print(pytorch_model_summary.summary(self.layer2, torch.zeros(32, 128+17), show_input=True))
#        print(pytorch_model_summary.summary(self.layer2, torch.zeros(32, 128+17), show_input=False))

        conv, linear = _generate_conv_model(), _generate_linear_model()
        self._value_branch_1 = nn.Sequential(*conv).float()
        self._value_branch_2 = nn.Sequential(*linear).float()

        self._value_branch_final = SlimFC(
            in_size=256,#prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        ).float()

#        print(pytorch_model_summary.summary(self._value_branch, torch.zeros(32,256), show_input=True))

        self._features = None


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        character, terrain = input_dict['obs'].values() 

        self._features = (character, terrain)

        out_1 = self.layer1(terrain.float())
        out_2 = self.layer2(torch.cat((out_1, character), 1).float())

        # hold the current output for the value function()
#        self._features = out_2

        out = self._final_layer(out_2)
        return out, state 


    @override(TorchModelV2)
    def value_function(self):
#        assert self._features is not None, "must call forward() first"
        
#        out = self._value_branch(self._features).squeeze(1)

        character, terrain = self._features
        out_1 = self._value_branch_1(terrain.float())
        out_2 = self._value_branch_2(torch.cat((out_1, character), 1).float())
        out = self._value_branch_final(out_2)
        return out.squeeze(1)
