from dataclasses import dataclass
from typing import Optional


@dataclass
class Experiment:
    name: str
    seed: int
    output_dir: str


@dataclass
class Data:
    root: str
    download: bool
    num_workers: int


@dataclass
class Diffusion:
    T: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    sigma_type: str
    predict: str


@dataclass
class Model:
    base_channels: int
    channel_mult: list[int]
    num_res_blocks: int
    attention_resolutions: list[int]
    dropout: float
    time_embed_dim: int


@dataclass
class Training:
    batch_size: int
    shuffle: bool
    lr: float
    grad_clip: float
    num_steps: int
    ema_decay: float
    mixed_precision: bool
    checkpoint_every: int


@dataclass
class Logging:
    log_every: int
    sample_every: int
    num_samples: int
    show_steps_every: int


@dataclass
class Config:
    experiment: Experiment
    data: Data
    diffusion: Diffusion
    model: Model
    training: Training
    logging: Logging
