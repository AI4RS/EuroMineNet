import os
import numpy as np
from PIL import Image
from glob import glob
import cv2
def reconstruct_label_from_png_patches(patch_dir, original_shape, crop_size=128, step_size=64):
    """
    Reconstruct the original label from overlapping PNG patches.

    Args:
        patch_dir (str): Directory containing PNG label patches.
        original_shape (tuple): Original label shape as (H, W).
        crop_size (int): Size of each patch (assumed square).
        step_size (int): Step size used in cropping.
    
    Returns:
        np.ndarray: Reconstructed label as a (H, W) uint8 array.
    """

    H, W = original_shape
    recon = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    patch_idx = 0

    h_steps = list(range(0, H - crop_size + 1, step_size))
    w_steps = list(range(0, W - crop_size + 1, step_size))

    # Top-left grid
    for h in h_steps:
        for w in w_steps:
            patch = cv2.imread(os.path.join(patch_dir, f"Patch{patch_idx}.png"), cv2.IMREAD_GRAYSCALE)
            recon[h:h+crop_size, w:w+crop_size] = patch
            patch_idx += 1

    if H % step_size != 0:
        h = H - crop_size
        for w in w_steps:
            patch = cv2.imread(os.path.join(patch_dir, f"Patch{patch_idx}.png"), cv2.IMREAD_GRAYSCALE)
            recon[h:h+crop_size, w:w+crop_size] = patch
            patch_idx += 1

    # Right edge
    if W % step_size != 0:
        w = W - crop_size
        for h in h_steps:
            patch = cv2.imread(os.path.join(patch_dir, f"Patch{patch_idx}.png"), cv2.IMREAD_GRAYSCALE)
            recon[h:h+crop_size, w:w+crop_size] = patch
            patch_idx += 1

        # Bottom edge


    # Bottom-right corner
    if H % step_size != 0 and W % step_size != 0:
        h = H - crop_size
        w = W - crop_size
        patch = cv2.imread(os.path.join(patch_dir, f"Patch{patch_idx}.png"), cv2.IMREAD_GRAYSCALE)
        recon[h:h+crop_size, w:w+crop_size] = patch

    # Convert back to 0/255 uint8 format
    return np.rint(recon).astype(np.uint8)

if __name__=="__main__":
    #segmentation results:
    # root="./test_results/"
    # save_root="./segmentation_results_reconstructed/"
    # reference_root="./MineNetCDV2_Cropped160_Step160/"
    # sample_file_suffix="label/Year2015.tif"
    # models=os.listdir(root)
    # for model in models:
    #     domains=os.listdir(os.path.join(root, model))
    #     for domain in domains:
    #         years=os.listdir(os.path.join(root, model, domain))
    #         full_path_to_sample=os.path.join("MineNetCDV2", domain, sample_file_suffix)
    #         sample_shape=np.array(Image.open(full_path_to_sample))
    #         os.makedirs(os.path.join(save_root, model, domain), exist_ok=True)
    #         for year in years:
    #             reconstructed_file=reconstruct_label_from_png_patches(patch_dir=os.path.join(root, model, domain, year), original_shape=sample_shape.shape, crop_size=160, step_size=160)
    #             reconstructed_image=Image.fromarray(reconstructed_file)
    #             reconstructed_image.save(os.path.join(save_root, model, domain, f"Year{year}.png"))
    #cd results:
    root="./results_uni/MNCDV2"
    save_root="./cd_results_reconstructed_Uni/"
    reference_root="./MineNetCDV2_Cropped160_Step160/"
    sample_file_suffix="label/Year2015.tif"
    models=os.listdir(root)
    for model in models:
        domains=os.listdir(os.path.join(root, model))
        for domain in domains:
            years=os.listdir(os.path.join(root, model, domain))
            full_path_to_sample=os.path.join("MineNetCDV2", domain, sample_file_suffix)
            sample_shape=np.array(Image.open(full_path_to_sample))
            os.makedirs(os.path.join(save_root, model, domain), exist_ok=True)
            for year in years:
                reconstructed_file=reconstruct_label_from_png_patches(patch_dir=os.path.join(root, model, domain, year), original_shape=sample_shape.shape, crop_size=160, step_size=160)
                reconstructed_image=Image.fromarray(reconstructed_file)
                reconstructed_image.save(os.path.join(save_root, model, domain, f"Year{year}.png"))
    

    # domain="Bulgaria_1"
    
    # patch_path="test_results/deeplabv3_plus/Bulgaria_1/2015"

    # full_path_to_sample=os.path.join("MineNetCDV2", domain, sample_file_suffix)

    
    # print(sample_shape.shape)
    # reconstructed_file=reconstruct_label_from_png_patches(patch_dir=patch_path, original_shape=sample_shape.shape, crop_size=160, step_size=160)

    # reconstructed_image=Image.fromarray(reconstructed_file)
    # reconstructed_image.save("sample1.png")