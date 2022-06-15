from dataclasses import dataclass


@dataclass
class Config:
    wandb_project = 'Effective_dl_4'
    device: str
    grad_accum_steps: int = 32
    batch_size: int = 2
    clip_grad_norm: float = 3.
    overfit_batch:bool = False

    def get(self, attr, default_value=None):
        return getattr(self, attr, default_value)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)

        raise KeyError(key)

