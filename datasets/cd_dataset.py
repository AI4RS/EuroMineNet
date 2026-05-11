from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms as tfs
import torch
import itertools
import numpy as np

class change_detection_dataset_local(Dataset):
    def __init__(self,path, transform) -> None:
        super().__init__()
        self.pre_change_path=os.path.join(path,"A")
        self.post_change_path=os.path.join(path,"B")
        self.change_gt_path=os.path.join(path,"label")
        self.fname_list=os.listdir(self.pre_change_path)
        self.transform=transform
    def __getitem__(self, index):
        fname=self.fname_list[index]
        pre_img=Image.open(os.path.join(self.pre_change_path,fname)).convert("RGB")
        post_img=Image.open(os.path.join(self.post_change_path,fname)).convert("RGB")
        change_gt=Image.open(os.path.join(self.change_gt_path,fname)).squeeze().long()
        # transform=transforms.Compose([
        #     transforms.ToTensor()
        # ])
        pre_tensor=self.transform(pre_img)
        post_tensor=self.transform(post_img)
        gt_tensor=self.transform(change_gt)
        return {'pre':pre_tensor,'post':post_tensor,'gt':gt_tensor,'fname':fname}
    def __len__(self):
        return len(self.fname_list)
    
class change_detection_dataset_HG(Dataset):
    def __init__(self,dataset,transform=None) -> None:
        super().__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return(len(self.dataset))
    def __getitem__(self, index):
        imageA=self.transform(self.dataset[index]["imageA"])
        imageB=self.transform(self.dataset[index]["imageB"])
        label=tfs.ToTensor()(self.dataset[index]["label"]).squeeze().long()
        return {'pre':imageA,'post':imageB,'gt':label,'fname':str(index)+".png"}
    

from train_val_test_splits import train_val_test_splits

normalization={"mean": [1089.74398414, 1004.04424665,  912.91169804, 1157.54372767, 1956.3718531, 2314.40816642, 2249.62494194, 2551.81568494, 1920.88963249, 1185.54458012] 
 ,"std": [220.01112071, 258.62166282, 385.29996291, 352.84198064, 433.08390432, 555.78649618, 556.49755492, 616.86119128, 615.74838491, 531.51408023]}

class Bitemporal_Change_Detection_Dataset(Dataset):
    def __init__(self, dataset_root, data_format='npy', year_range=(2015,2024), mode="train", normalization=normalization):
        super().__init__()

        self.data_format=data_format

        self.dataset_root=dataset_root
        self.domains=train_val_test_splits[mode]
        self.normalization=normalization

        self.start_year, self.end_year = year_range[0], year_range[1]

        # get file path using one year
        files_paths=[os.listdir(os.path.join(dataset_root, domain, str(self.start_year), "image")) for domain in self.domains]
        domains_files_paths=dict(zip(self.domains, files_paths))

        year_range=list(range(self.start_year,self.end_year+1,1))
        self.combinations=list(itertools.combinations(year_range, 2))
        
        self.samples=[]
        for domain in self.domains:
            for _, (year1, year2) in enumerate(self.combinations):
                assert year1<year2, f"year1 {year1} should less than year2 {year2}"
                for patch in domains_files_paths[domain]:
                    patch_num=patch.split(".")[-2].split("_")[-1]
                    combination=(domain, year1, year2, patch_num)
                    self.samples.append(combination)

    # name format of combi files f'.../{domain}_{year1}_{year2}_{patch_num}.npy'
    def __getitem__(self, index):
        (domain, year1, year2, patch_num)= self.samples[index]

        image1_path=os.path.join(self.dataset_root, domain, str(year1), "image", f'Year{year1}_{patch_num}.npy')
        image2_path=os.path.join(self.dataset_root, domain, str(year2), "image", f'Year{year2}_{patch_num}.npy')
        label1_path=os.path.join(self.dataset_root, domain, str(year1), "label", f'Year{year1}_{patch_num}.npy')
        label2_path=os.path.join(self.dataset_root, domain, str(year2), "label", f'Year{year2}_{patch_num}.npy')

        image1=np.load(image1_path)
        image2=np.load(image2_path)
        label1=np.load(label1_path)
        label2=np.load(label2_path)
        
        assert len(label1.shape)==2, f'expected label1 to have 1 channel but got {label1.shape[0]} (loaded from {label1_path})'
        assert len(label2.shape)==2, f'expected label2 to have 1 channel but got {label2.shape[0]} (loaded from {label2_path})'
        # assert len(image1.shape)==3, f'expected label1 to have 1 {image1_path}'
        # assert len(image2.shape)==3, f'expected label1 to have 1 {label1_path}'

        label1[label1!=0]=1
        label2[label2!=0]=1

        label1=torch.from_numpy(label1).long()
        label2=torch.from_numpy(label2).long()


        # combination_label=label2-label1+1 # 0 deconstruction 1 unchanged 2 Newly-constructed
        combination_label=torch.abs(label2-label1) # 0 unchanged 1 changed

        # combination_label=label2-label1
        # combination_label[combination_label==-1]=0 # 0 unchanged 1 expansion

        image1=torch.from_numpy(image1).float()
        image2=torch.from_numpy(image2).float()

        if self.normalization is not None:
            image1=tfs.Normalize(mean=self.normalization["mean"], std=self.normalization["std"])(image1)
            image2=tfs.Normalize(mean=self.normalization["mean"], std=self.normalization["std"])(image2)

        return {"pre":image1, "post": image2, "gt": combination_label, "fname": f'{domain}_{year1}_{year2}_{patch_num}'}
        return {"images":[image1, image2], "labels": [label1, label2], 'CD_label': combination_label, 'Metadata': {"domain": domain, "year1": year1, "year2": year2, "patchnum": patch_num}}

    def __len__(self):
        return len(self.samples)


if __name__=="__main__":
    # doamin_dict=get_num_patches_per_domain("/home/eric/dataset/MineNetCDV2_Cropped128_Step64")
    # splits=split_dataset_by_sites(doamin_dict, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=4321)
    # print(splits)

    # dataset=Combinations_Change_Detection_Dataset("/home/eric/dataset/MineNetCDV2_Cropped128_Step64", mode="train", normalization=normalization)

    dataset=Bitemporal_Change_Detection_Dataset("/Users/weikangyu/Documents/MNCDV2/MineNetCDV2_Cropped160_Step160", mode="train", year_range=(2015,2024), normalization=normalization)
    dataloader_train=DataLoader(dataset, batch_size=8,num_workers=16,shuffle=True)
    # for i, batch in enumerate(dataloader_train):
    #     print(batch["images"][2015].shape)
        # for year, image in batch["images"]:
        #     print(year, image.shape)
        # print(batch["image"].min())
    print(len(dataset))

    # print(dataset[0])
    # data= next(enumerate(dataloader_train))

    for i, data in enumerate(dataloader_train):
        print(data)

    # print(len(data[1]["labels"]))