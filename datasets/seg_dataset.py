import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import itertools
import torchvision.transforms as tfs

"""
Train Sites: 92, Patches: 838530
Validation Sites: 26, Patches: 119565
Test Sites: 18, Patches: 240210
min values per band are [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
max values per band are [5918. 5789. 6049. 6049. 6287. 6767. 6767. 6767. 6731. 6731.]
"""
from train_val_test_splits import train_val_test_splits

normalization={"mean": [1089.74398414, 1004.04424665,  912.91169804, 1157.54372767, 1956.3718531, 2314.40816642, 2249.62494194, 2551.81568494, 1920.88963249, 1185.54458012] 
 ,"std": [220.01112071, 258.62166282, 385.29996291, 352.84198064, 433.08390432, 555.78649618, 556.49755492, 616.86119128, 615.74838491, 531.51408023]}

class Multitemporal_Change_Detection_Dataset(Dataset):
    def __init__(self, dataset_root, data_format='npy', year_range=range(2015,2024+1), mode="train", normalization=normalization):
        super().__init__()
        self.data_format=data_format

        self.dataset_root=dataset_root
        self.domains=train_val_test_splits[mode]
        self.normalization=normalization

        # self.start_year, self.end_year = year_range[0], year_range[1]
        self.year_range=list(year_range)
        # get file path using one year
        files_paths=[os.listdir(os.path.join(dataset_root, domain, str(self.year_range[0]), "image")) for domain in self.domains]
        domains_files_paths=dict(zip(self.domains, files_paths))

        
        self.combinations=list(itertools.combinations(self.year_range, 2))

        self.samples=[]
        for domain in self.domains:
            for patch in domains_files_paths[domain]:
                patch_num=patch.split(".")[-2].split("_")[-1]
                combination=(domain, patch_num)
                self.samples.append(combination)

    def __getitem__(self, index):
        domain, patch_num = self.samples[index]

        images_over_years=[]

        for year in self.year_range:
            image=np.load(os.path.join(self.dataset_root, domain, str(year), 'image', f'Year{year}_{patch_num}.npy'))
            image_ts=torch.from_numpy(image).float()
            if self.normalization is not None:
                image_ts=tfs.Normalize(mean=self.normalization["mean"], std=self.normalization["std"])(image_ts)
            images_over_years.append(image_ts)
            # print(self.normalization)
            
    
        labels_over_years=[]
        # images_to_return=dict(zip(list(self.year_range), images_over_years))

        for year in self.year_range:
            label=np.load(os.path.join(self.dataset_root, domain, str(year), 'label', f'Year{year}_{patch_num}.npy'))
            label_ts=torch.from_numpy(label).long()
            label_ts[label_ts!=0]=1
            labels_over_years.append(label_ts.long())

        # print(len(labels_over_years))
        # labels_to_return=dict(zip(list(self.year_range), labels_over_years))

        # combinations_year1_year2=[]
        # combinations_labels=[]
        # for _, (year1, year2) in enumerate(self.combinations):
        #     assert year1<year2, f"year1 {year1} should less than year2 {year2}"
        #     combinations_year1_year2.append(f'{year1}_{year2}')
        #     label_year1, label_year2=labels_to_return[year1], labels_to_return[year2]
        #     combination_label=label_year2-label_year1+1 # 0 deconstruction 1 unchanged 2 Newly-constructed
        #     combinations_labels.append(combination_label)

        # combinations_labels_to_return=dict(zip(combinations_year1_year2, combinations_labels))

        return {"images": images_over_years, "labels": labels_over_years}
        # return {"images": images_to_return, "labels": labels_to_return, "combinations_labels": combinations_labels_to_return}

    def __len__(self):
        return len(self.samples)
    
class Multitemporal_Semantic_Segmentation_Dataset(Dataset):
    def __init__(self, dataset_root, data_format='npy', year_range=range(2015,2024+1), mode="train", normalization=normalization):
        super().__init__()
        self.data_format=data_format

        self.dataset_root=dataset_root
        self.domains=train_val_test_splits[mode]
        self.normalization=normalization

        # self.start_year, self.end_year = year_range[0], year_range[1]
        self.year_range=list(year_range)
        # get file path using one year
        files_paths=[os.listdir(os.path.join(dataset_root, domain, str(self.year_range[0]), "image")) for domain in self.domains]
        domains_files_paths=dict(zip(self.domains, files_paths))

        self.samples=[]
        for domain in self.domains:
            for patch in domains_files_paths[domain]:
                for year in self.year_range:
                    patch_num=patch.split(".")[-2].split("_")[-1]
                    combination=(domain, year, patch_num)
                    self.samples.append(combination)

    def __getitem__(self, index):
        domain, year, patch_num = self.samples[index]


        image=np.load(os.path.join(self.dataset_root, domain, str(year), 'image', f'Year{year}_{patch_num}.npy'))
        image_ts=torch.from_numpy(image).float()
        if self.normalization is not None:
            image_ts=tfs.Normalize(mean=self.normalization["mean"], std=self.normalization["std"])(image_ts)

        label=np.load(os.path.join(self.dataset_root, domain, str(year), 'label', f'Year{year}_{patch_num}.npy'))
        label_ts=torch.from_numpy(label).long()
        label_ts[label_ts!=0]=1

        return {"images": image_ts, "labels": label_ts, "domain": domain, "year": year, "patch_num": patch_num}

    def __len__(self):
        return len(self.samples)
        

def get_num_patches_per_domain(dataset_root):

    start_year, end_year = 2015, 2024
    domains=os.listdir(dataset_root)
    # get file path using one year
    files_paths=[os.listdir(os.path.join(dataset_root, domain, str(start_year), "image")) for domain in domains]
    domains_files_paths=dict(zip(domains, files_paths))

    year_range=list(range(start_year,end_year+1,1))
    combinations=list(itertools.combinations(year_range, 2))

    domain_counts=[]
    for domain in domains:
        domain_count=0
        for _, (year1, year2) in enumerate(combinations):
            assert year1<year2, f"year1 {year1} should less than year2 {year2}"
            for patch in domains_files_paths[domain]:
                patch_num=patch.split(".")[-2].split("_")[-1]
                combination=(domain, year1, year2, patch_num)
                domain_count+=1
        domain_counts.append(domain_count)
    domain_dict=dict(zip(domains, domain_counts))
    return domain_dict
import random

def split_dataset_by_sites(site_patches, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle sites randomly
    sites = list(site_patches.keys())
    random.shuffle(sites)
    
    # Compute total number of patches
    total_patches = sum(site_patches.values())
    train_target = total_patches * train_ratio
    val_target = total_patches * val_ratio
    test_target = total_patches * test_ratio
    
    # Allocate sites to splits based on patch count
    train_sites, val_sites, test_sites = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    
    for site in sites:
        patches = site_patches[site]
        if train_count + patches <= train_target:
            train_sites.append(site)
            train_count += patches
        elif val_count + patches <= val_target:
            val_sites.append(site)
            val_count += patches
        else:
            test_sites.append(site)
            test_count += patches
    
    print(f"Train Sites: {len(train_sites)}, Patches: {train_count}")
    print(f"Validation Sites: {len(val_sites)}, Patches: {val_count}")
    print(f"Test Sites: {len(test_sites)}, Patches: {test_count}")
    
    return {
        "train": train_sites,
        "val": val_sites,
        "test": test_sites
    }

if __name__=="__main__":
    # doamin_dict=get_num_patches_per_domain("/home/eric/dataset/MineNetCDV2_Cropped128_Step64")
    # splits=split_dataset_by_sites(doamin_dict, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=4321)
    # print(splits)

    # dataset=Combinations_Change_Detection_Dataset("/home/eric/dataset/MineNetCDV2_Cropped128_Step64", mode="train", normalization=normalization)

    # dataset=Multitemporal_Change_Detection_Dataset("/home/eric/dataset/MineNetCDV2_Cropped160_Step160", mode="train", year_range=range(2015,2025,1), normalization=normalization)

    dataset=Multitemporal_Semantic_Segmentation_Dataset("/home/weikang/datasets/MineNetCDV2_Cropped160_Step160", mode="train", year_range=range(2015,2025,1), normalization=normalization)
    dataloader_train=DataLoader(dataset, batch_size=8,num_workers=16,shuffle=True)
    # for i, batch in enumerate(dataloader_train):
    #     print(batch["images"][2015].shape)
        # for year, image in batch["images"]:
        #     print(year, image.shape)
        # print(batch["image"].min())
    print(len(dataset))
    data= next(enumerate(dataloader_train))

    print((data[1]["images"].shape, data[1]["labels"].shape))