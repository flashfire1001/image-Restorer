# adding noise to the clear tensor x.

# write a class to add noise to the original image and return corrupted image in batches
import torch

class NoiseGenerator:
    def __init__(self, eta=0.56, sp_amount=0.1, block_size=10):
        super().__init__()
        self.eta = eta
        self.sp_amount = sp_amount
        self.block_size = block_size
        self.corruptions = ['gaussian', 'saltpepper', 'block']

    def add_gaussian_noise(self, image):
        noise = torch.randn_like(image,device = image.device)
        return image + self.eta * noise

    def add_sp_noise(self, image):
        noisy = image.clone()
        mask = torch.rand_like(noisy,device = noisy.device)
        noisy[mask < self.sp_amount / 2] = 0.0 # nearly self.sp_amount / 2 * num_image pixels or set to 0
        noisy[(mask >= self.sp_amount / 2) & (mask < self.sp_amount)] = 1.0
        return noisy

    def add_block_mask(self, image):
        masked = image.clone()
        C, H, W = image.shape
        # use randint to generate a point of the mask rectangular 
        x = torch.randint(0, H - self.block_size, (1,)).item()
        y = torch.randint(0, W - self.block_size, (1,)).item()
        masked[:, x: x+self.block_size, y: y+self.block_size] = 0.0
        return masked

    def sample(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: Tensor of shape (B, C, H, W)
        returns: corrupted Tensor (B, C, H, W)
        """
        B, C, H, W = images.shape
        device = images.device
        corrupted_images = []

        # xs = torch.randint(0, H - self.block_size, (B,), device = device)
        # ys = torch.randint(0, W - self.block_size, (B,), device = device)
        # for i in range(B):
        #     x, y = xs[i], ys[i]
        #     images[i, :, x:x + self.block_size, y:y + self.block_size] = 0.0

        # return images
        
        for i in range(B):
            img = images[i]
            corrupted = self.add_gaussian_noise(img)
            corrupted_images.append(corrupted)
            
        # for i in range(B):
        #     img = images[i]
        #     corruption_type = self.corruptions[torch.randint(0, len(self.corruptions), (1,)).item()]
        #     if corruption_type == "gaussian":
        #         corrupted = self.add_gaussian_noise(img)
        #     elif corruption_type == "saltpepper":
        #         corrupted = self.add_sp_noise(img)
        #     elif corruption_type == "block":
        #         corrupted = self.add_block_mask(img)
        #     corrupted_images.append(corrupted)

        return torch.stack(corrupted_images)

    def sample_guassian_noise(images: torch.Tensor):
        return torch.randn_like(images,device = images.device)