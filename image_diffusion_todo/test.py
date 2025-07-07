# smoke_test.py (put this in image_diffusion_todo/)

import torch
from scheduler import DDPMScheduler
from model import DiffusionModule
from network import UNet

# instantiate the scheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_1=1e-4,
    beta_T=0.02,
    mode="linear",
    sigma_type="small"
)

# test the forward and reverse step
x = torch.randn(2, 3, 64, 64)            # two dummy images
t = torch.tensor([10, 500])              # two timesteps
x_t, noise = scheduler.add_noise(x, t)   # forward noising
x_prev = scheduler.step(x_t, t[0].item(), noise)  # one reverse step

# instantiate your unet
net = UNet()  

# wrap in the DiffusionModule
dm = DiffusionModule(network=net, var_scheduler=scheduler)

# test loss
batch = torch.randn(4, 3, 64, 64)        # batch of 4 dummy images
loss = dm.get_loss(batch)                
loss.backward()

print(" Smoke test passed - everything seems to wrok")
