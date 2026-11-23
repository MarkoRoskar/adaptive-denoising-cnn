from matplotlib.pylab import Enum
from generate_noise import noise_add, noise_mul
from generate_image import generate_image
import torch
import numpy as np
import random
    
class NoiseType(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

def generate_dataset(dataset_size, noise_type):
    print("Generating images....")
    original_images = []
    noisy_images= []
    original_images = [generate_image() for _ in range(dataset_size)]
    if noise_type == NoiseType.ADDITIVE:
        noisy_images = [noise_add(img, std=random.uniform(0.1, 0.3)) for img in original_images]
    else:
        noisy_images = [noise_mul(img, std=random.uniform(0.1, 0.3)) for img in original_images]
    original_images = [(img / 255.0).astype(np.float32) for img in original_images]
    noisy_images = [(img / 255.0).astype(np.float32) for img in noisy_images]
    
    # convert images to pytorch tensors
    original_images = torch.tensor(np.array(original_images))
    noisy_images = torch.tensor(np.array(noisy_images))
    original_images = original_images.permute((0, 3, 1, 2))
    noisy_images = noisy_images.permute((0, 3, 1, 2))
    dataset_file_name = f"train_dataset_{'add' if noise_type == NoiseType.ADDITIVE else 'mult'}.pt"
    
    print(f"Saving dataset in {dataset_file_name} file")
    # save original & noisy tensors
    torch.save({
        "original_images": original_images,
        "noisy_images": noisy_images
    }, dataset_file_name)

    print("Dataset saved")
    
    return