import numpy as np

def noise_add(img, std):
    # Generate Gaussian noise
    noise = np.random.normal(0, std * 255, img.shape).astype(np.int16)
    
    # Add the noise to the image and clip the values to stay within valid range [0, 255]
    img_noisy = img.astype(np.int16) + noise
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    return img_noisy

def noise_mul(img, std):
    # Generate random noise
    noise = np.random.normal(loc=0, scale=std, size=img.shape).astype(np.float32)
    # Apply multiplicative noise
    noisy_image = img.astype(np.float32) * (1 + noise)
    
    # Clip the values to stay in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image