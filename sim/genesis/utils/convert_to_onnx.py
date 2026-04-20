import torch
import torch.nn as nn
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to ONNX format')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file (.pkl)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.pt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ONNX file path')
    return parser.parse_args()

args = parse_args()

# 加载配置
with open(args.cfg, 'rb') as f:
    cfgs = pickle.load(f)
    print(f"Loaded config: {cfgs}")

# 加载模型
checkpoint = torch.load(args.model, map_location=torch.device("cpu"))
model_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
obs_norm_state_dict = checkpoint.get("obs_norm_state_dict")

# 创建继承自ActorCritic的模型类
from rsl_rl.modules import ActorCritic

class ExportModel(ActorCritic):
    def __init__(self, *model_args, obs_mean=None, obs_std=None, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        if obs_mean is not None and obs_std is not None:
            self.register_buffer("obs_mean", obs_mean)
            self.register_buffer("obs_std", obs_std)
        else:
            self.obs_mean = None
            self.obs_std = None

    def forward(self, obs):
        if self.obs_mean is not None and self.obs_std is not None:
            obs = (obs - self.obs_mean) / torch.clamp(self.obs_std, min=1e-6)
        return self.actor(obs)

# 根据配置创建模型实例        
model = ExportModel(
    num_actor_obs=cfgs[1]['num_single_obs'],
    num_critic_obs=cfgs[1]['num_single_obs'],
    num_actions=cfgs[0]['num_actions'],
    actor_hidden_dims=cfgs[4]['policy']['actor_hidden_dims'],
    critic_hidden_dims=cfgs[4]['policy']['critic_hidden_dims'],
    activation='elu',
    init_noise_std=cfgs[4]['policy']['init_noise_std'],
    obs_mean=obs_norm_state_dict["_mean"] if obs_norm_state_dict else None,
    obs_std=obs_norm_state_dict["_std"] if obs_norm_state_dict else None,
)

# 加载模型参数
model.load_state_dict(model_state_dict)
model.eval()

# 创建示例输入
obs_dim = cfgs[1]['num_single_obs']  # 从obs_cfg获取观测维度
dummy_input = torch.randn(1, obs_dim)

# 转换为ONNX
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=['obs'],
    output_names=['actions'],
    dynamic_axes={
        'obs': {0: 'batch_size'},
        'actions': {0: 'batch_size'}
    }
)

print("Model successfully converted to ONNX format")
