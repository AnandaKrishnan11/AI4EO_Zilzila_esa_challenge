from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, transform
import numpy as np
import os
from glob import glob
import torch


def get_train_val_index(N, split_ratio):
    n_tr = int(split_ratio*N)
    n_vs = N-n_tr

    v_inds = list(range(0, N, int(N/n_vs)))
    tr_inds = list(set(list(range(0, N))).symmetric_difference(set(v_inds)))

    return tr_inds, v_inds


def systematic_split(pre, post, label, split_ratio):
    assert len(pre) == len(post) == len(label), 'The pre, post and label length is not the same'

    N = len(pre)
    tind, vind = get_train_val_index(N=N, split_ratio=split_ratio)

    pre_t, pre_v = [pre[ind] for ind in tind], [pre[ind] for ind in vind]
    post_t, post_v = [post[ind] for ind in tind], [post[ind] for ind in vind]
    lbl_t, lbl_v = [label[ind] for ind in tind], [label[ind] for ind in vind]

    return pre_t, post_t, lbl_t, pre_v, post_v, lbl_v



class DamageDataset(Dataset):
    def __init__(self, pre, post, lbl):
        super().__init__(self, DamageDataset)
        
        self.pre = pre
        self.post = post
        self.lbl = lbl


    def __len__(self):
        return len(self.pre)
    

    def __getitem__(self, index):
        img_pre = self.pre[index]
        img_post = self.post[index]
        lbl = self.lbl[index]

        pre = self.resize_image(imread(img_pre))
        post = self.resize_image(imread(img_post))
        lbl = self.read_txt(lbl)

        pre = self.transform_image(image=pre)
        post = self.transform_image(image=post)
        lbl = self.transform_txt(num=lbl)

        pre, post, lbl


    def resize_image(self, image, height, width):
        trs_img = transform.resize(image, 
                                (height, width, image.shape[-1]),  
                                order=0,                 
                                preserve_range=True,      
                                anti_aliasing=False       
                                )
        trs_img = trs_img.astype(image.dtype)
        return trs_img


    def read_txt(self, txt):
        with open(txt, 'r') as my_txt:
            out = int(my_txt.read_lines()[0])
        return out
    
    def make_normal(self, image, mu=None, std=None):
        if mu is not None:
            image = (image - mu)/std
        else:
            image = (image - image.min())/(image.max()-image.min() + 1e-6)
        
        return image

    def transform_image(self,image):
        image = np.moveaxis(image, 0, -1)
        image = self.make_normal(image=image)
        image = torch.from_numpy(image)
        return image
    
    def transform_txt(self, num):
        num = torch.from_numpy(num) # given this is the discrete data point 
        return num



def get_dataloader(data_dir, split_ratio):

    pre_t = []
    post_t = []
    lbl_t = []

    pre_v = []
    post_v = []
    lbl_v = []

    for fold in os.listdir(data_dir):
        pre_images = sorted(glob(f'{data_dir}/{fold}/pre_event_crops/*.tif'))
        post_images = sorted(glob(f'{data_dir}/{fold}/post_event_crops/*.tif'))
        lbls_images = sorted(glob(f'{data_dir}/{fold}/labels/*.txt'))

        a,b,c, d, e, f = systematic_split(pre=pre_images, post=post_images, label=lbls_images, split_ratio=split_ratio)

        pre_t += a
        post_t += b
        lbl_t += c

        pre_v += d
        post_v += e
        lbl_v += f

    train_dataset = DamageDataset(pre=pre_t, post=post_t, lbl=lbl_t)
    valid_dataset = DamageDataset(pre=pre_v, post=post_v, lbl=lbl_v)

    train_loader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=True, drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=12, shuffle=True, drop_last=False)

    return train_loader, valid_loader
    

