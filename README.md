# Mean-Flow-image-restore-project

A minimal implementation of a image-restorer based on MNIST dataset.

Originally copied from: https://github.com/haidog-yaqub/MeanFlow

Some latest updates:

- [x] include UNet into model
- [ ] do experiment: test the denoising effect, train with cfg, while generating with/without class-info.
- [ ] do experiment: train with/without cfg (class = None)
- [x] do experiment: train 200k - 300k with cosine annealing scheduler, log the PSNR and lr and loss, batch enlarged dim,headers and shrink the batch size(64) . (I encountered CUDA collapse recursively after 150k batches. It crashed for no reason. Failed)
- [ ] do experiment: train with block-mask; sp-noise corrupted image. for 300k.
- [ ] do experiment: train with different guidance-scale(2, 3, 5)
- [x] try UNet for image restoration of Guassian noise.(I encountered NaN for my losses and I can't overcome the issue. Failed)
- [ ] try latent space representation in train_latent.py

If you are interested in this project, train it yourself and notify me! The codes are well commented for your reuse. 
