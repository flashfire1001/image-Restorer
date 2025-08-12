# do metric evaluation and visualize them in graph.
# smoothed loss,mse + loss,mse 
# PSNR + smoothed psnr.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
import torch


class MetricTracker:
    def __init__(self, window_length= 21, polyorder=3):
        assert window_length % 2 == 1, "window_length must be odd"
        self.window_length = window_length
        self.polyorder = polyorder

        self.losses = []
        self.mses = []
        self.psnrs = []  

    def update(self, loss: float, mse: float):
        """record the mse and loss(adaptive L2) loss """
        self.losses.append(loss)
        self.mses.append(mse)
        
    def update_from_list(self, losses:list, mses :list):
        self.losses = losses
        self.mses = mses
        
    def _smooth(self, values):
        if len(values) < self.window_length:
            return values
        return savgol_filter(values, self.window_length, self.polyorder)


    def calc_psnr(self, original_image: torch.Tensor, restored_image: torch.Tensor):
        """Calculate the PSNR between the original and restored image."""
        pixel_error = original_image - restored_image
        mse = (pixel_error ** 2).mean()

        # Ensure it's on CPU and convert to float before using NumPy
        mse_value = mse.detach().cpu().item()
        psnr = 10 * np.log10(1.0 / (mse_value + 1e-8))
        self.psnrs.append(psnr)
    
    def plot(self, save_path: Path = None):
        steps = np.arange(len(self.losses))
        num_psnr = np.arange(len(self.psnrs))
        smoothed_losses = self._smooth(self.losses)
        smoothed_mses = self._smooth(self.mses)
        smoothed_psnrs = self._smooth(self.psnrs)
        
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        # Plot Loss
        axs[0].plot(steps, self.losses, label="Loss", alpha=0.5)
        axs[0].plot(steps, smoothed_losses, label="Smoothed Loss", color='red')
        axs[0].set_title("Loss over Time")
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)
        
        #Plot MSE
        axs[1].plot(steps, self.mses, label="MSE", alpha=0.5)
        axs[1].plot(steps, smoothed_mses, label="Smoothed MSE", color='red')
        axs[1].set_title("MSE over Time")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("MSE")
        axs[1].legend()
        axs[1].grid(True)
    
        # Plot PSNRs
        axs[2].plot(num_psnr * 500, self.psnrs, label="PSNR", alpha=0.5)
        axs[2].plot(num_psnr * 500, smoothed_psnrs, label="Smoothed PSNR", color='green')
        axs[2].set_title("PSNR over Time")
        axs[2].set_xlabel("Step")
        axs[2].set_ylabel("PSNR (dB)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    
