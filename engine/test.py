import argparse
from glob import glob
import os
from skimage.io import imread, transform
import numpy as np
import torch
import pandas as pd
import geopandas as gpd

from models import create_model


def preprocess_input(x, width, height, mu=None, std=None): # this process the input the same way as training data processed
    x = transform.resize(x, 
                        (height, width, x.shape[-1]),  
                        order=0,                 
                        preserve_range=True,      
                        anti_aliasing=False       
                        )
    x = x.astype(x.dtype)

    if mu is not None:
        x = (x - mu)/std
    else:
        x = (x - x.min())/(x.max()-x.min() + 1e-6)
    
    x = np.moveaxis(x, 0, -1)
    x = torch.from_numpy(x)
    return x.unsqueeze(0)  # batched input


def getargs():
    pars = argparse.ArgumentParser(description='test argument parser')
    pars.add_argument('--data_dir', type=str, help='root data directory')
    pars.add_argument('--height', type=str, help='image height interms of pixels to resize', default=128)
    pars.add_argument('--width', type=str, help='image width interms of pixels to resize', default=128)
    pars.add_argument('--model_type', type=str, help='model type either resnet or combined')
    pars.add_argument('--num_class', type=int, help='number of damage classes')
    pars.add_argument('--pretrained', action='store_false')
    pars.add_argument('--resnet_depth', type=int, help='ResNet model depth either 50 or 101', default=50)
    pars.add_argument('--model_path', type=str, help='path to save the model', default='/')
    pars.add_argument('--save_dir', type=str, help='path to save the prediction values', default='/')
    pars.add_argument('--shape_path', type=str, help='shapefile path to add predicted values as attribute')
    return pars.parse_args()
    


def test(width, height, num_class, model_type, depth, pretrained, model_path, save_dir, shape_path, data_dir):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=num_class, model_type=model_type, depth=depth, pretrained=pretrained)
    model.load_state_dict(torch.load(f'{model_path}/best_weight.pt'))  # this we can play with either best weight or last weight
    model = model.to(device)
    model.eval()

    for fold in os.listdir(data_dir):

        ids = []
        cls = []

        pre_images = sorted(glob(f'{data_dir}/{fold}/pre_event_crops/*.tif'))
        post_images = sorted(glob(f'{data_dir}/{fold}/post_event_crops/*.tif'))

        files = list(zip(pre_images, post_images))

        for pairs in files:
            pre_im = preprocess_input(pairs[0], width=width, height=height)
            post_im = preprocess_input(pairs[1], width=width, height=height)

            pre_im = pre_im.to(device)
            post_im = post_im.to(device)

            logits = model(pre_im, post_im)
            class_probs = torch.softmax(logits, dim=1)
            class_indices = torch.argmax(class_probs, dim=1)

            ids.append(os.path.split(pairs[0])[1][:-4])  # the id files are saved
            cls.append(class_indices.squeeze(0).detach().cpu().numpy().item())  # this finaly converts to 0 or 1
        dicts = {'id':ids, 'cls':cls}
        df = pd.DataFrame.from_dict(dicts)

        df.to_csv(f'{save_dir}/{fold}.csv')
        

        gdf = gpd.read_file(f'{shape_path}/{fold}.shp')  # This assumes in each shapefile is saved in a shapefile path with folder name

        gdf['damaged'] = cls  # assuming all rows have valid geometry, else we have to perfom table join

        gdf.to_file(f'{save_dir}/{fold}.shp')



def main(args):
    test(width=args.width,
         height=args.height,
         num_classes=args.num_class,
         model_type=args.model_type,
         depth=args.resnet_depth,
         pretrained=args.pretrained,
         model_path=args.model_path,
         save_dir=args.save_dir,
         shape_path=args.shape_path,
         data_dir=args.data_dir)
    


if __name__ == '__main__':
    args = getargs()
    main(args=args)