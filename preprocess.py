# Data Preprocessing for MineNetCD V2
# import cv2
import rasterio
import os
import numpy as np
import itertools
from PIL import Image
import pandas as pd
from torchvision.utils import save_image
import torch

dataset_path='./MineNetCDV2'
domains=os.listdir(dataset_path)
excluded_domains=["Spain_4", "Sweden_6", "Poland_16"]
domains=[i for i in domains if i not in excluded_domains]

# domains=["Greece_4"]
# domains=list(set(domains)-set(['Greece_11', 'Finland_6', 'Slovakia_1', 'Hungary_1'])) # For dubugging Finland_6 missing 1 pixel for some labels
year_range=list(range(2015,2025,1))

def load_sentinel2(path):
    src=rasterio.open(path)
    np_data=src.read()
    src.close()
    return np_data

def load_label(path):
    label=Image.open(path)
    label_array=np.array(label)
    return label_array

def padding_with_repetition(image, size):
    print(image.shape, "padding")
    # size should be at least 2 times of h/w
    if len(image.shape)==2:
        h, w = image.shape
    elif len(image.shape)==3:
        c, h, w = image.shape
    pad_h, pad_w=size-h, size-w
    if pad_h>0 and pad_w>0:
        if len(image.shape)==2:
            image_temp=np.zeros((size, size), dtype=image.dtype)
            image_temp[:h, :w]= image
            image_temp[h:, :w]= image[:size-h, :w]
            image_temp[:h, w:]= image[:h, :size-w]
            image_temp[h:, w:]= image[:size-h, :size-w]
        elif len(image.shape)==3:
            image_temp=np.zeros((c, size, size), dtype=image.dtype)
            image_temp[:, :h, :w]= image
            image_temp[:, h:, :w]= image[:, :size-h, :w]
            image_temp[:, :h, w:]= image[:, :h, :size-w]
            image_temp[:, h:, w:]= image[:, :size-h, :size-w]
    elif pad_h>0:
        if len(image.shape)==2:
            image_temp=np.zeros((size, w), dtype=image.dtype)
            image_temp[:h, :]= image
            image_temp[h:, :]= image[:size-h, :]
        elif len(image.shape)==3:
            image_temp=np.zeros((c, size, w), dtype=image.dtype)
            image_temp[:, :h, :]= image
            image_temp[:, h:, :]= image[:, :size-h, :]
    elif pad_w>0:
        if len(image.shape)==2:
            image_temp=np.zeros((h, size), dtype=image.dtype)
            image_temp[:, :w]= image
            image_temp[:, w:]= image[:, :size-w]
        elif len(image.shape)==3:
            image_temp=np.zeros((c, h, size), dtype=image.dtype)
            image_temp[:, :, :w]= image
            image_temp[:, :, w:]= image[:, :, :size-w]
    else:
        raise ValueError("No need for Padding")
    return image_temp


def calculate_mean_std(dataset_path, domains, year_range):
    #mean [1089.11031607, 1003.96145353,  912.98432349, 1157.79532432, 1957.07361426, 2315.29489318, 2250.45729051, 2552.64732003, 1922.21076081, 1186.55621542] 
    #std [219.94873173, 258.82652232, 385.68401713, 353.12406873, 433.0992919, 555.77189576, 556.4700335, 616.88989387, 615.78610617, 531.69222223]
    # domains=list(set(domains)-set(['Poland_17']))
    values_per_band=np.zeros(10)
    mins_per_band=np.zeros(10)
    maxs_per_band=np.zeros(10)
    num_pixels=0
    for domain in domains:
        work_dir=os.path.join(dataset_path, domain)
        for year in year_range:
            current_image=load_sentinel2(os.path.join(work_dir, 'image', f'Year{year}.tif'))
            n_channels, height, width = current_image.shape
            num_pixels+= height*width

            min_value=np.min(current_image, axis=0)[0]
            max_value=np.max(current_image, axis=0)[0]

            for i in range(len(mins_per_band)):
                mins_per_band[i]=min_value[i] if min_value[i]<mins_per_band[i] else mins_per_band[i]
                maxs_per_band[i]=max_value[i] if max_value[i]>maxs_per_band[i] else maxs_per_band[i]
        
        

            values_per_band += np.sum(current_image, axis=(1,2))
            # else:
            #     # print(values_per_band.shape,current_image.reshape(10,-1).shape)
            #     values_per_band=np.append(values_per_band,current_image.reshape(10,-1), axis=1)
            #     # values_per_band=np.stack([values_per_band, current_image.reshape(10,-1)], axis=0)
            #     # print(values_per_band.shape)
    mean = values_per_band/num_pixels
    print("the mean value is", mean, "min values per band are", mins_per_band, "max values per band are", maxs_per_band)

    values_per_band_std=np.zeros(10)
    for domain in domains:
        work_dir=os.path.join(dataset_path, domain)
        for year in year_range:
            current_image=load_sentinel2(os.path.join(work_dir, 'image', f'Year{year}.tif'))
            n_channels, height, width = current_image.shape
            num_pixels+= height*width

            values_per_band_std += np.sum(np.square(current_image-mean[:,np.newaxis,np.newaxis]), axis=(1,2))

    print('mean', mean, '\n', 'std', (values_per_band_std/num_pixels)**0.5)

def get_combinations(year_range):
    combinations=list(itertools.combinations(year_range, 2))
    return combinations

def rename(dataset_path, domains, year_range, mode):
    for domain in domains:
        work_dir=os.path.join(dataset_path, domain)
        for year in year_range:
            filepath=os.path.join(work_dir, mode, f'Label{year}.tif')

            if os.path.isfile(filepath):
                os.rename(filepath, filepath.replace('Label', 'Year'))

            filepath=os.path.join(work_dir, mode, f'year{year}.tif')

            if os.path.isfile(filepath):
                os.rename(filepath, filepath.replace('year', 'Year'))

def check_label(dataset_path, domains, year_range, mode):
    for domain in domains:
        work_dir=os.path.join(dataset_path, domain)
        for year in year_range:
            filepath=os.path.join(work_dir, mode, f'Year{year}.tif')
            file=Image.open(filepath)
            array=np.unique(np.array(file))

            example=np.unique(np.array([0, 255]))
            if not (array==example).all():
                print(array)
    print("all values have been checked, they all contain only 0/255")

def check_shape(dataset_path, domains, year_range):
    mismatched_domains=[]
    for domain in domains:
        mismatched=0
        work_dir=os.path.join(dataset_path, domain)
        for year in year_range:

            current_image=load_sentinel2(os.path.join(work_dir, "image", f'Year{year}.tif'))
            current_label=load_label(os.path.join(work_dir, "label", f'Year{year}.tif'))

            if current_image.shape[1:]!=current_label.shape:
                print(f'size mismatched between image and label for {domain}_{year}, image size {current_image.shape}, label size {current_label.shape}')
                mismatched=1
        if mismatched==1:
            mismatched_domains.append(domain)
    print('all files have been checked, mismatched_domains found in', mismatched_domains)

def crop_and_save(dataset_path, domains, year_range, save_root, crop_size, step_size):
    total_images_cropped=0
    for domain in domains:
        work_dir=os.path.join(dataset_path, domain)
        save_dir_root=os.path.join(save_root,domain)
        for year in year_range:
            save_dir_image=os.path.join(save_dir_root, str(year), "image")
            save_dir_label=os.path.join(save_dir_root, str(year), "label")
            os.makedirs(save_dir_image, exist_ok=True)
            os.makedirs(save_dir_label, exist_ok=True)

            current_image=load_sentinel2(os.path.join(work_dir, "image", f'Year{year}.tif'))
            current_label=load_label(os.path.join(work_dir, "label", f'Year{year}.tif'))

            if current_image.shape[1]<crop_size or current_image.shape[2]<crop_size:
                current_image=padding_with_repetition(current_image, crop_size)
                current_label=padding_with_repetition(current_label, crop_size)

            assert current_image.shape[1:]==current_label.shape, f'size mismatched between image and label for {domain}_{year}, image size {current_image.shape}, label size {current_label.shape}'
            
            # print(os.path.join(work_dir, mode, f'Year{year}.tif'))
            
            patch_num_image=crop_and_save_patches(save_dir_image, f'Year{year}', crop_size, step_size, current_image)
            patch_num_label=crop_and_save_patches_label(save_dir_label, f'Year{year}', crop_size, step_size, current_label)

            assert patch_num_image==patch_num_label, f'size mismatched between image and label for {domain}_{year}, image size {current_image.shape}, label size {current_label.shape}'

        total_images_cropped+=patch_num_image
        # print(current_label.shape, current_image.shape,patch_num_image, domain)
    
    print('amount of cropped samples in one year:', total_images_cropped, '\n', 'amount of all samples in 10 years:', total_images_cropped*10)

def crop_and_save_patches(save_path, suffix_name, crop_size, step_size, image, mean=None, std=None):
    n_channels, height, width = image.shape
    h_patches, w_patches=(height-crop_size)//step_size, (width-crop_size)//step_size
    patch_num=0
    # print(height, width, crop_size, step_size, h_patches, w_patches)
    # print(image.shape, h_patches, w_patches)
    for h in range(h_patches+1):
        for w in range(w_patches+1):
            image_cropped=image[:, h*step_size:h*step_size+crop_size, w*step_size:w*step_size+crop_size]

            assert h*step_size+crop_size<=height and w*step_size+crop_size<=width
            
            save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            np.save(sample_save_path, image_cropped)
            patch_num=patch_num+1
    if height%crop_size!=0:
        for w in range(w_patches+1):
            image_cropped=image[:, height-crop_size: height, w*step_size:w*step_size+crop_size]
            save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            np.save(sample_save_path, image_cropped)
            patch_num=patch_num+1
    if width%crop_size!=0:
        for h in range(h_patches+1):
            image_cropped=image[:, h*step_size:h*step_size+crop_size, width-crop_size: width]
            save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            np.save(sample_save_path, image_cropped)
            patch_num=patch_num+1
    if height%crop_size!=0 and width%crop_size!=0:
        image_cropped=image[:, height-crop_size: height, width-crop_size: width]
        save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
        sample_save_path=os.path.join(save_path, save_name)
        if mean is not None and std is not None:
            image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
        np.save(sample_save_path, image_cropped)
        patch_num=patch_num+1
    return patch_num

#amount of cropped samples in one year: 23563 #amount of all samples in 10 years: 235630
def crop_and_save_patches_label(save_path, suffix_name, crop_size, step_size, image):
    # print(image.shape)
    height, width = image.shape
    h_patches, w_patches=(height-crop_size)//step_size, (width-crop_size)//step_size
    patch_num=0
    # print(image.shape, h_patches, w_patches)
    for h in range(h_patches+1):
        for w in range(w_patches+1):
            image_cropped=image[h*step_size:h*step_size+crop_size, w*step_size:w*step_size+crop_size]

            assert h*step_size+crop_size<=height and w*step_size+crop_size<=width
            
            save_name=f'Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)

            image_cropped=image_cropped.astype(np.float32)
            save_image(torch.from_numpy(image_cropped), sample_save_path.replace('.npy', '.png'))
            patch_num=patch_num+1
    if height%crop_size!=0:
        for w in range(w_patches+1):
            image_cropped=image[height-crop_size: height, w*step_size:w*step_size+crop_size]
            save_name=f'Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            image_cropped=image_cropped.astype(np.float32)
            save_image(torch.from_numpy(image_cropped), sample_save_path.replace('.npy', '.png'))
            patch_num=patch_num+1
    if width%crop_size!=0:
        for h in range(h_patches+1):
            image_cropped=image[h*step_size:h*step_size+crop_size, width-crop_size: width]
            save_name=f'Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            image_cropped=image_cropped.astype(np.float32)
            save_image(torch.from_numpy(image_cropped), sample_save_path.replace('.npy', '.png'))
            patch_num=patch_num+1
    if height%crop_size!=0 and width%crop_size!=0:
        image_cropped=image[height-crop_size: height, width-crop_size: width]
        save_name=f'Patch{str(patch_num)}.npy'
        sample_save_path=os.path.join(save_path, save_name)
        image_cropped=image_cropped.astype(np.float32)
        save_image(torch.from_numpy(image_cropped), sample_save_path.replace('.npy', '.png'))
        patch_num=patch_num+1
    return patch_num


def process_matching_labels(dataset_path, domains, year_range, save_root):
    combinations=get_combinations(year_range)
    total_count=0
    domain_counts=[]
    domains_ordered=[]
    domain_counts_constructed=[]
    domain_counts_unchanged=[]
    domain_counts_deconstructed=[]
    for domain in domains:
        work_dir=os.path.join(dataset_path, domain)
        save_dir_root=os.path.join(save_root,domain)
        os.makedirs(save_dir_root, exist_ok=True)
        domain_count=0
        domain_count_constructed=0
        domain_count_unchanged=0
        domain_count_deconstructed=0
        for _, (year1, year2) in enumerate(combinations):
            assert year1<year2, f"prechange year {year1} should be earlier than postchange year {year2}"
            year1_path=os.path.join(work_dir, str(year1), 'label')
            year2_path=os.path.join(work_dir, str(year2), 'label')
            patch_list=os.listdir(year1_path)
            for patch in patch_list:
                patch_year2=patch.replace(str(year1), str(year2))
                patch_num=patch.split(".")[-2].split("_")[-1]
                label1=np.load(os.path.join(year1_path, patch))
                label2=np.load(os.path.join(year2_path, patch_year2))
                label1[label1==255]=1
                label2[label2==255]=1
                matched_label=label2-label1
                matched_label=matched_label+1 # 0 deconstruction 1 unchanged 2 Newly-constructed
                uniques, counts= np.unique(matched_label, return_counts=True)
                counts=dict(zip(uniques, counts))
                # print(counts)
                for k, v in counts.items():
                    if k==0:
                        domain_count_deconstructed+=v
                    elif k==1:
                        domain_count_unchanged+=v
                    elif k==2:
                        domain_count_constructed+=v

                suffix_name=f'{domain}_{year1}_{year2}_{patch_num}.npy'
                np.save(os.path.join(save_dir_root, suffix_name), matched_label)
                domain_count=domain_count+1
                total_count=total_count+1
        domain_counts.append(domain_count)
        domains_ordered.append(domain)
        domain_counts_constructed.append(domain_count_constructed)
        domain_counts_deconstructed.append(domain_count_deconstructed)
        domain_counts_unchanged.append(domain_count_unchanged)

    stats=pd.DataFrame({"domains": domains_ordered, "domain_counts": domain_counts, "domain_counts_constructed":domain_counts_constructed, 
                        "domain_counts_deconstructed": domain_counts_deconstructed, "domain_counts_unchanged": domain_counts_unchanged})
    stats.to_csv(os.path.join(save_root, 'stats.csv'), index = False)
    print("generated", total_count, "labels for multitemporal CD.")
if __name__=="__main__":
    # path='/home/eric/dataset/MineNetCDV2/Austria_1/image/Year2015.tif'
    # src=load_sentinel2(path)
    # print(src.shape, type(src))
    # mean, std= np.mean(src, axis=(1,2)), np.std(src, axis=(1,2))
    # print(mean.shape, std.shape, mean, std)
    # print(np.sum(src, axis=(1,2)))
    # print(src.reshape(10,-1).shape)
    # mean=np.ones((10))
    # print(src.shape, mean.shape)
    # print(src-mean[:,np.newaxis,np.newaxis])
    # print(np.sum(np.square(np.subtract(src-mean)), axis=(1,2)).shape)
    # calculate_mean_std(dataset_path, domains, year_range)


    crop_and_save(dataset_path, domains, year_range, './MineNetCDV2_Cropped160_Step160', 160, 160)

    # check_shape(dataset_path, domains, year_range)
    # process_matching_labels('/home/eric/dataset/MineNetCDV2_Cropped128_Step64/', domains, year_range, '/home/eric/dataset/MineNetCDV2_Cropped128_Step64_matched_labels/')
    # rename(dataset_path, domains, year_range, 'label') # some filenames are ambiguous, this is to unify the filenames

    # check_label(dataset_path, domains, year_range, 'label')