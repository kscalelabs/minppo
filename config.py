"""Defines the model configuration options."""

import sys
from dataclasses import dataclass, field

from omegaconf import MISSING, OmegaConf


@dataclass
class EnvironmentConfig:
    n_frames: int = field(default=1)
    backend: str = field(default="mjx")
    include_c_vals: bool = field(default=True)


@dataclass
class VisualizationConfig:
    camera_name: str = field(default=MISSING)
    width: int = field(default=640)
    height: int = field(default=480)
    render_every: int = field(default=1)
    max_steps: int = field(default=1000)
    video_length: float = field(default=5.0)
    video_save_path: str = field(default="episode.mp4")
    num_episodes: int = field(default=20)


@dataclass
class RewardConfig:
    termination_height: float = field(default=-0.2)
    height_min_z: float = field(default=-0.2)
    height_max_z: float = field(default=2.0)
    is_healthy_reward: float = field(default=5)
    original_pos_reward_exp_coefficient: float = field(default=2)
    original_pos_reward_subtraction_factor: float = field(default=0.2)
    original_pos_reward_max_diff_norm: float = field(default=0.5)
    ctrl_cost_coefficient: float = field(default=0.1)
    weights_ctrl_cost: float = field(default=0.1)
    weights_original_pos_reward: float = field(default=4)
    weights_is_healthy: float = field(default=1)
    weights_velocity: float = field(default=1.25)


@dataclass
class ModelConfig:
    hidden_size: int = field(default=256)
    num_layers: int = field(default=2)
    activation: str = field(default="tanh")


@dataclass
class TrainingConfig:
    lr: float = field(default=3e-4)
    num_envs: int = field(default=2048)
    num_steps: int = field(default=10)
    total_timesteps: int = field(default=1_000_000_000)
    update_epochs: int = field(default=4)
    num_minibatches: int = field(default=32)
    gamma: float = field(default=0.99)
    gae_lambda: float = field(default=0.95)
    clip_eps: float = field(default=0.2)
    ent_coef: float = field(default=0.0)
    vf_coef: float = field(default=0.5)
    max_grad_norm: float = field(default=0.5)
    anneal_lr: bool = field(default=True)
    normalize_env: bool = field(default=True)
    physics_n_frames: int = field(default=1)
    debug: bool = field(default=True)


@dataclass
class InferenceConfig:
    model_path: str = field(default=MISSING)


@dataclass
class Config:
    kscale_id: str = field(default=MISSING)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    debug: bool = field(default=True)


def load_config_from_cli() -> Config:
    args = sys.argv[1:]
    if len(args) < 1:
        raise ValueError("Usage: cli <config_path> (<additional_args> ...)")
    path, *other_args = args
    raw_config = OmegaConf.load(path)
    config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, raw_config)
    if other_args:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(other_args))
    return config
