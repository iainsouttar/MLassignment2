# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 08:02:12 2018

@author: iains
"""
from skimage.morphology import label
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage
import numpy as np
from glob import glob
from skimage.transform import resize
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import imageio
from skimage.io import imread


data_dir=os.path.join('/modules/cs342/', 'Assignment2') #Data directory to be used

print "Loading image"

all_images = glob(os.path.join(data_dir, 'Full*','*', '*', '*'))
image_df = pd.DataFrame({'path': all_images}) #create dataframe of paths of all images


#Add columns corresponding to image key, whether it's a mask or original, and whether it's train or test
image_df['ImageId'] = image_df['path'].map(lambda path: path.split('/')[-3])
image_df['ImageType'] = image_df['path'].map(lambda path: path.split('/')[-2])
image_df['TrainingSplit'] = image_df['path'].map(lambda path: path.split('/')[-4])

print "Images loaded"

test_df = image_df.query('TrainingSplit=="FullTesting"')

IMG_WIDTH = 256; IMG_HEIGHT = 256; #Standard size of image


test_img_df = test_df.loc[test_df['ImageType']=='images'] #dataframe of images

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # Flatten
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


    
import pandas as pd

def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    #Convert image to grayscale
    image_id = im_path.split('/')[-3]
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)
    
    #Produce mask using threshold
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    
    
    #swap if image is 'light'
    if (test_img_df.loc[test_img_df['ImageId']==image_id]['Class'].iloc[0]=='Light'):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
        #bels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': image_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list):
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df

df = analyze_list_of_images(list(test_img_df['path']))
df.to_csv('thresholding.csv', index=None)




