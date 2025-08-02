from datasets import get_mnist_dataloader
from models import MFDiT
import torch
from meanflow import MeanFlow
from utils import load_checkpoint
from torchvision.utils import make_grid,save_image
from matplotlib import pyplot as plt
import numpy as np

test_dataloader = get_mnist_dataloader(batch_size= 10)
device = "cuda" if torch.cuda.is_available() else "cpu"
project_name = "MF_image_restorer"


# create and config the model, optimizer instance for training
model = MFDiT(
    input_size=32,
    patch_size=2,
    in_channels=1,
    dim=72,
    depth=6,
    num_heads=3,
    num_classes=10,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

# set the CFG training parameters, meanflow is a set of functions for meanflow cfg training
meanflow = MeanFlow(channels=1,
                    image_size=32,
                    num_classes=10,
                    flow_ratio=0.50,
                    time_dist=['lognorm', -0.4, 1.0],
                    cfg_ratio=0.10,
                    cfg_scale=2.0,
                    # experimental
                    cfg_uncond='u')

model = load_checkpoint(model = model, project_name= project_name, trained_step = 100000)


data, z, noise = meanflow.sample_each_class(model, 1) 
imgs_tensor = torch.cat((data, z, noise),dim = 0)
log_img = make_grid(imgs_tensor, nrow=10)
img_save_path = f"test.png"
save_image(log_img, img_save_path)
print("image generated and saved, displaying...")
np_img = log_img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]           
plt.imshow(np.clip(np_img, 0, 1), cmap = "grey")
plt.axis("off")
plt.show()
