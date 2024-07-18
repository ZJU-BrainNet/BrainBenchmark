from dataclasses import dataclass


@dataclass
class ModelArgs:
    d_model: int = 2560
    patch_len: int = 1024
    seq_len: int = 16

    time_n_layers: int = 8
    channel_n_layers: int = 2
    n_heads: int = 8

    norm_eps: float = 1e-7
    ff_hidden: int = d_model*3

    predict_mode: str = 'linear'

    drop_prob: float = 0.1
    learnable_mask: bool = False
    mask_ratio: float = 0.4


