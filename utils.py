import torch
from pathlib import Path
from torchinfo import summary

class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def load_checkpoint(model:torch.nn.Module, project_name:str, trained_step:int, root_dir = "checkpoints",):
    """load the model with pretrained params of trained_step"""
    parent_path = Path(root_dir) 
    checkpoint_path = parent_path / project_name / f"step_{trained_step}.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"model loaded from {checkpoint_path} to {next(model.parameters()).device}")
    return model


MiB = 1024 ** 2

def model_size_mib(model: torch.nn.Module) -> float:
    # calc the size of a model
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size / MiB

def simulate(noise:torch.Tensor, batch_size,  model:torch.nn.Module) -> torch.Tensor:
    """use euler simulation to generate a set of images"""