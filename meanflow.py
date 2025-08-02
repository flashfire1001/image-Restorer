import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
from utils import Normalizer
from noise import NoiseGenerator
from datasets import get_mnist_dataloader

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    train more on easier samples, while penalize the hard ones(outliers or too noisy)
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # square of the error
    return (stopgrad(w) * loss).mean() # more flexible than standard mse.
    


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        #by default minmax(with out latent space.)
        normalizer=['minmax', None, None],
        # mean flow settings, the rate of r = t.
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        # ratio of masked class label.
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.noise_generator = NoiseGenerator()

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size * 0.5)
        indices_1 = np.random.permutation(batch_size)[num_selected:2*num_selected]
        indices_2 = np.random.permutation(batch_size)[0:num_selected]
     
        r_np[indices_1] = t_np[indices_1]
        t_np[indices_2] = r_np[indices_2]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        # create t,r and arrange their elements in correct shape.(for broadcasting)
        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        x = self.normer.norm(x)
        #e = torch.randn_like(x) # simple function for generating epsilon.
        e = self.noise_generator.sample(x)
        

        z = (1 - t_) * x + t_ * e # sample the z_t which is the position in midst.
        v = e - x # calc the value of velocity field at (z_t)

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            # w is the cfg scale. 
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond) # use the model t = t to generate the average instantaneous velocity at location z , time_step t.
                v_hat = self.w * v + (1 - self.w) * u_t # this is v(z_t,t|c)
                
                # another way to deal with the cfg, different from the paper.
                if self.cfg_uncond == 'v':
                    # offical JAX repo uses original v for unconditional items
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                #otherwise directly use v for there is no class info.
                v_hat = v

        # forward pass
        # u = model(z, t, r, y=c)
        # this is a smart way to devoid of the integer class label which can't take gradient.
        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        
        # for both l2(mse) and adaptive loss.
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda', ):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class) # create a (0,1,...,num_classes-1) with n_per_class repetition
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        dataloader = get_mnist_dataloader(batch_size = c.shape[0], train = False, num_workers = 1)
        data, _ = next(iter(dataloader))
        data = data.to(device)
        z =  self.noise_generator.sample(data)
        noise = z.clone()

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device) # t_intervals

        # print(t_vals)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}"
            
            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            uncond = torch.ones_like(c) * self.num_classes
            v = model(z , t,  r, uncond)
            z = z - (t_-r_) * v # z is the raw image tensor.
            
        z = self.normer.unnorm(z) # the denoised corrupted image
        return data, z, noise