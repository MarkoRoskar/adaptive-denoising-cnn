import torch
from torch.utils.data import DataLoader, TensorDataset
from generate_noise import noise_add, noise_mul
from generate_image import generate_image
from generate_dataset import  NoiseType
import numpy as np
import math
import random

class FilterBranch(torch.nn.Module):
    def __init__(self, n_filts=8) -> None:
        super().__init__()
        self.filter_conv_r = torch.nn.Conv2d(in_channels=1, out_channels=n_filts, kernel_size=11, padding=5, bias=False)
        self.filter_conv_g = torch.nn.Conv2d(in_channels=1, out_channels=n_filts, kernel_size=11, padding=5, bias=False)
        self.filter_conv_b = torch.nn.Conv2d(in_channels=1, out_channels=n_filts, kernel_size=11, padding=5, bias=False)
    
    def forward(self, x): # [N, 3, 256, 256]
        r, g, b = torch.chunk(x, chunks=3, dim=1) # 3 x [N, 1, 256, 256]

        r_filters = self.filter_conv_r(r) # [N, n_filts, 256, 256]
        g_filters = self.filter_conv_g(g) # [N, n_filts, 256, 256]
        b_filters = self.filter_conv_b(b) # [N, n_filts, 256, 256]

        return r_filters, g_filters, b_filters # 3 x [N, n_filts, 256, 256]
    
class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels=32, out_channels = 32, dropout_rate=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.dropout1 = torch.nn.Dropout2d(p=dropout_rate)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.dropout2 = torch.nn.Dropout2d(p=dropout_rate)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu2(x)

        return x

class ClasificationBranch(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, n_filts=8):
        super().__init__()
        self.resNetBlock_1 = ResNetBlock(3, 32, dropout_rate)
        self.resNetBlock_2 = ResNetBlock(32, 32, dropout_rate)
        self.resNetBlock_3 = ResNetBlock(32, 32, dropout_rate)
        
        self.classify_conv = torch.nn.Conv2d(in_channels=32, out_channels=n_filts, kernel_size=1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x_res = self.resNetBlock_1(x)
        x_res = self.resNetBlock_2(x_res)
        x_res = self.resNetBlock_3(x_res)
        
        class_maps = self.classify_conv(x_res)

        class_maps = self.softmax(class_maps)

        return class_maps
    
class EPSFilterNet(torch.nn.Module):
    def __init__(self, n_filts=8) -> None:
        super().__init__()
        self.filterBranch = FilterBranch(n_filts)
        self.classificationBranch = ClasificationBranch(dropout_rate=0.5, n_filts=n_filts)
        
    def forward(self, x):       # [N, 3, 256, 256]
        r_filters, g_filters, b_filters = self.filterBranch(x) # 3 x [N, n_filts, 256, 256]
        
        class_maps = self.classificationBranch(x)
        weighted_r = torch.sum(r_filters * class_maps, dim=1, keepdims=True)
        weighted_g = torch.sum(g_filters * class_maps, dim=1, keepdims=True)
        weighted_b = torch.sum(b_filters * class_maps, dim=1, keepdims=True)

        output = torch.cat((weighted_r, weighted_g, weighted_b), dim=1)
        output = torch.clamp(output, 0, 1)
        
        return output
    
def nn_train(model, optimizer, loss_fn, epochs, noise_fn, batch_size, device):
    train_dataset_path=""
    if noise_fn.__name__ == "noise_add":
        train_dataset_path = "train_dataset_add.pt"
    else:
        train_dataset_path = "train_dataset_mult.pt"
        
    # Load train dataset
    train_data = torch.load(train_dataset_path, weights_only=True)
    
    original_images = train_data["original_images"]
    noisy_images = train_data["noisy_images"]
    #print(f"Images: shape - {original_images.shape}, type - {original_images.dtype}")
    #print(f"Noisy images: shape - {noisy_images.shape}, type - {noisy_images.dtype}")
    use_cuda=True
    if device != "cuda":
        use_cuda=False
    dataset = TensorDataset(noisy_images, original_images)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        loss_sum=0.0
        for inputs_batch, targets_batch in data_loader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()

            # forward pass - get predictions from the model
            outputs = model(inputs_batch)

            # compute loss between original images (targets) & noisy images (inputs)
            loss = loss_fn(outputs, targets_batch)

            # backward pass - compute gradients & update model weights
            loss.backward()
            optimizer.step()

            # accumulate training loss for each batch
            loss_sum += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {loss_sum / len(data_loader)}")
    print("Training finished.")

def calculate_snr(noisy_img, original_img):
    noisy_img = (noisy_img/255.0).astype(np.float32)
    original_img = (original_img/255.0).astype(np.float32)
    noise = noisy_img - original_img
    signal_power = np.mean(original_img ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_psnr(noisy_img, original_img):
    noisy_img = (noisy_img/255.0).astype(np.float32)
    original_img = (original_img/255.0).astype(np.float32)
    mse = np.mean((noisy_img - original_img) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def nn_test(model, noise_type, device):
    originial_image=generate_image()
    noisy_image=[]
    denoised_image=[]
    std=random.uniform(0.1, 0.3)
    if noise_type == NoiseType.ADDITIVE:
        noisy_image=noise_add(originial_image, std=std)
    else:
        noisy_image=noise_mul(originial_image, std=std)
    originial_image_c=(originial_image/255.0).astype(np.float32)
    noisy_image_c=(noisy_image/255.0).astype(np.float32)
    originial_image_c=originial_image_c[..., np.newaxis]
    noisy_image_c=noisy_image_c[..., np.newaxis]
    originial_image_c=originial_image_c.transpose((3, 2, 0, 1))
    noisy_image_c=noisy_image_c.transpose((3, 2, 0, 1))
    # print(f"Image: shape - {originial_image_c.shape}, type - {originial_image_c.dtype}")
    # print(f"Noisy_image: shape - {noisy_image_c.shape}, type - {noisy_image_c.dtype}")
    with torch.no_grad():
        image = torch.from_numpy(noisy_image_c).to(device)
        pred=model(image)
        pred_np = pred.cpu().detach().numpy()  # Convert to NumPy
        #print(f"Predicted_np_image: shape - {pred_np.shape}, type - {pred_np.dtype}")
        pred_np = np.squeeze(pred_np)
        #print(f"Predicted_np_image: shape - {pred_np.shape}, type - {pred_np.dtype}")
        pred_np = pred_np.transpose((1, 2, 0))
        #print(f"Predicted_np_image: shape - {pred_np.shape}, type - {pred_np.dtype}")
        pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
        #print(f"Predicted_np_image: shape - {pred_np.shape}, type - {pred_np.dtype}")
        denoised_image=pred_np
        
    snr_noisy = calculate_snr(noisy_image, originial_image)
    psnr_noisy = calculate_psnr(noisy_image, originial_image)
    snr_filt = calculate_snr(denoised_image, originial_image)
    psnr_filt = calculate_psnr(denoised_image, originial_image)
    return originial_image, noisy_image, denoised_image, snr_noisy, psnr_noisy, snr_filt, psnr_filt

def nn_eval(model, iterations, noise_type, device):
    snr_noisy_list = []
    psnr_noisy_list = []
    snr_filt_list = []
    psnr_filt_list = []

    for _ in range(iterations):
        _, noisy, denoised, snr_noisy, psnr_noisy, snr_filt, psnr_filt = nn_test(model, noise_type, device)
        snr_noisy_list.append(snr_noisy)
        psnr_noisy_list.append(psnr_noisy)
        snr_filt_list.append(snr_filt)
        psnr_filt_list.append(psnr_filt)

    # Convert to NumPy arrays for easier math
    snr_noisy_list = np.array(snr_noisy_list)
    psnr_noisy_list = np.array(psnr_noisy_list)
    snr_filt_list = np.array(snr_filt_list)
    psnr_filt_list = np.array(psnr_filt_list)

    # Print results with mean ± std
    print(f"SNR (noisy): {snr_noisy_list.mean():.2f} ± {snr_noisy_list.std():.2f} dB")
    print(f"PSNR (noisy): {psnr_noisy_list.mean():.2f} ± {psnr_noisy_list.std():.2f} dB")
    print(f"SNR (filtered): {snr_filt_list.mean():.2f} ± {snr_filt_list.std():.2f} dB")
    print(f"PSNR (filtered): {psnr_filt_list.mean():.2f} ± {psnr_filt_list.std():.2f} dB")