#train without dumping the image tensor into latent space.

from models import MFDiT, MFUNet
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
from pathlib import Path
from datasets import get_mnist_dataloader
from utils import model_size_mib, load_checkpoint
from metric import MetricTracker

# create the path to save the results

torch.cuda.empty_cache()
parent_path = Path.cwd()
project_name = "MF_image_restorer"
image_path = parent_path / "images" /project_name 
checkpoint_path = parent_path /"checkpoints" / project_name 
image_path.mkdir(parents=True, exist_ok = True)
checkpoint_path.mkdir(parents = True, exist_ok = True)




if __name__ == '__main__':
    # set training info
    n_steps = 300000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    accelerator = Accelerator(mixed_precision='fp16')
  
    # set up the datalaoder for training
    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = get_mnist_dataloader(batch_size= batch_size, num_workers= 8) 
    train_dataloader = cycle(train_dataloader)

    # create and config the model, optimizer instance for training
    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        dim=144,
        depth=6,
        num_heads=3,
        num_classes=10,
    ).to(accelerator.device)

    # model = MFUNet(in_channels= 1,
    #                channels= [64, 128, 256],
    #                num_residual_layers= 2,
    #                t_embed_dim= 64,
    #                y_embed_dim= 64,
    #                num_classes= 10).to(accelerator.device)
    
    # model = load_checkpoint(model = model, project_name= project_name, trained_step = 100000)

    print(f"size of the model:{model_size_mib(model):.3f}Mib")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)


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

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Scheduler: Cosine Annealing
    # T_max is the number of iterations (or epochs) for one cycle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # initialize the experiment metrics
    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 100
    sample_step = 500
    save_checkpoint_step = 5000
    
    tracker = MetricTracker()
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader) # one batch is loaded for training.
            # question: the next(iter) method and enumerate method: how many num_workers are used for loading the data?
            
            # c: class
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)
            
            # train with/without class-info when use cfg.
            #c = torch.ones_like(c) * 10

            # let model generate hte
            loss, mse_val = meanflow.loss(model, x, c)
            tracker.update(loss.item(), mse_val.item())
            
            
            accelerator.backward(loss) #use accelerator's backward method instead of loss.backward
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            #Question: what is the difference of loss and mse_loss
            losses += loss.item()
            mse_losses += mse_val.item() 

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    # After training log_step steps, do the recording.
                    #current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {int(global_step)}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{batch_info}    {loss_info}    {lr_info}\n'

                    with open('log.txt', mode='a') as n:
                        # append the new message into the training log.
                        n.write(log_message)

                    losses = 0.0
                    mse_losses = 0.0

            if global_step % sample_step == 0:
                
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    original_images, z, noise = meanflow.sample_each_class(model, 1) 
                    imgs_tensor = torch.cat((original_images, z, noise),dim = 0)
                    tracker.calc_psnr(original_image=original_images, restored_image = z)
                    log_img = make_grid(imgs_tensor, nrow=10)
                    img_save_path = image_path/f"step_{int(global_step)}.png"
                    save_image(log_img, img_save_path)
                                    
                accelerator.wait_for_everyone()
                model.train()
                    
            if global_step % save_checkpoint_step == 0 or global_step == n_steps:
                #between each save_checkpoint_step, save the model state dict.
                ckpt_save_path = checkpoint_path / f"step_{int(global_step)}.pt"
                accelerator.save(model_module.state_dict(), ckpt_save_path)
    
    
    tracker.plot("metrics.png")