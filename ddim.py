!pip install diffusers==0.3.0

from diffusers import DDPMPipeline

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-ema-celebahq-256")
image_pipe.to("cuda")

from diffusers import UNet2DModel

repo_id = "google/ddpm-church-256"
model = UNet2DModel.from_pretrained(repo_id).to("cuda")

import torch

torch.manual_seed(0)

noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
).to("cuda")
noisy_sample = noisy_sample
print(noisy_sample)

from diffusers import DDPMPipeline
from diffusers import UNet2DModel
import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from diffusers import DDPMScheduler
from diffusers import DDIMScheduler
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import cv2
import os

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-church-256")
image_pipe.to("cuda")
repo_id = "google/ddpm-church-256"
model = UNet2DModel.from_pretrained(repo_id).to("cuda")

torch.manual_seed(0)

noisy_sample = torch.randn(
            1, model.config.in_channels, model.config.sample_size, model.config.sample_size
            ).to("cuda")

# for file in os.listdir("/content/gdrive/MyDrive/Shadow Removal DDPM Project/RePaint/log/face_example/gt_masked2/"):
img = PIL.Image.open("/content/gdrive/MyDrive/Shadow Removal DDPM Project/RePaint/log/face_example/gt_masked2/" + file)
img = img.resize((256, 256))
img = torchvision.transforms.functional.pil_to_tensor(img).to("cuda")
img = torch.reshape(img, (1,3,256,256))

new_noisy_sample = 0.80 * noisy_sample + 0.20 * img

scheduler = DDIMScheduler.from_config(repo_id)
scheduler.set_timesteps(num_inference_steps=500)

final = img

def display_sample(sample, i):
        image_processed = sample.cpu().permute(0,2,3,1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.numpy().astype(np.uint8)
        cv2.imwrite("/content/gdrive/MyDrive/CVSamplesDDIM/" + file, image_processed[0])
        display(f"Image at step {i}")
        image_pil = PIL.Image.fromarray(image_processed[0])
        display(image_pil)

import tqdm

sample = new_noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        with torch.no_grad():
            residual = model(sample, t).sample

            sample = scheduler.step(residual, t, sample).prev_sample

            if (i + 1) % 10 == 0:
                display_sample(sample, i + 1)