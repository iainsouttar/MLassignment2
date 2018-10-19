
# coding: utf-8

# In[ ]:


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
import pathlib
import imageio
from skimage.io import imread
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# In[ ]:


data_dir=os.path.join('/modules/cs342/', 'Assignment2') #Data directory to be used

print "Loading image"

all_images = glob(os.path.join(data_dir, 'Full*','*', '*', '*'))
image_df = pd.DataFrame({'path': all_images}) #create dataframe of paths of all images


#Add columns corresponding to image key, whether it's a mask or original, and whether it's train or test
image_df['ImageId'] = image_df['path'].map(lambda path: path.split('/')[-3])
image_df['ImageType'] = image_df['path'].map(lambda path: path.split('/')[-2])
image_df['TrainingSplit'] = image_df['path'].map(lambda path: path.split('/')[-4])

print "Images loaded"

train_df = image_df.query('TrainingSplit=="FullTraining"')

IMG_WIDTH = 256; IMG_HEIGHT = 256; #Standard size of image that I will use for model

masks_df = train_df.loc[train_df['ImageType']=='masks'] #dataframe of masks

train_img_df = train_df.loc[train_df['ImageType']=='images'] #dataframe of images



#create column of masks for corresponding original images and store these in dataframe
masksList = []
for images in train_img_df['ImageId']:
    imMasksList = []
    idMasks = masks_df['path'].loc[masks_df['ImageId']==images]
    #print idMasks
    for mask in idMasks:
        imMasksList.append(resize(imageio.imread(str(mask)), (IMG_HEIGHT, IMG_WIDTH)))
    masksList.append(np.sum(imMasksList, 0))
    #print 'done'
train_img_df['masks'] = masksList
print 'masks done'


# In[ ]:


print 'Loading all training image data to the dataframe'

IMG_CHANNELS = 3

train_img_df['images'] = train_img_df['path'].map(lambda x: imageio.imread(str(x))[:,:,:IMG_CHANNELS]/float(255))
print 'images done'
train_img_df['Grey'] = train_img_df['images'].map((lambda x: 0.21*x[:,:,0] + 0.72*x[:,:,1] + 0.07*x[:,:,2]))
train_img_df['Greyscale'] = train_img_df['images'].map(lambda x: (rgb2gray(x)))
train_img_df['Resized Grey'] = train_img_df['Greyscale'].map(lambda x: resize(x, (IMG_HEIGHT, IMG_WIDTH)))
print 'Resized Grey'
train_img_df['Original Size'] = train_img_df['images'].map((lambda x: (x.shape[:2])))

#create column of resized masks for corresponding original images and store these in dataframe
train_img_df['Class'] = train_img_df['images'].map(lambda x: "Light" if np.mean(x[:,:,2]) + np.mean(x[:,:,0]) > 0.8 else "Dark")
print "Completed"


# In[ ]:


#create training data and targets
X = []
t = []


for index, image in train_img_df.iterrows():
    tempIm = image["Resized Grey"].tolist()
    tempMask = image['masks'].tolist()
    for i in range(0, 4):
        X.append(np.rot90(tempIm, k=i))
        t.append(np.rot90(tempMask, k=i))


# In[ ]:


model = Sequential()
model.add(Dense(1024, input_shape=(256, 256)))
model.add(Dense(1024,activation='elu'))
model.add(Dropout(0, 15))
model.add(Dense(1024,activation='elu'))
model.add(Dropout(0, 15))

model.add(Dense(512,activation='elu'))
model.add(Dropout(0, 15))



model.add(Dense(512,activation='elu'))

model.add(Dense(256, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(np.asarray(X), t, validation_split = 0.1, nb_epoch = 30,
          batch_size=32)


# In[ ]:


test_img_df = image_df.query('TrainingSplit=="FullTesting"')
IMG_WIDTH = 256; IMG_HEIGHT = 256;
IMG_CHANNELS = 3

test_img_df['images'] = test_img_df['path'].map(lambda x: imageio.imread(str(x))[:,:,:IMG_CHANNELS]/float(255))
test_img_df['Greyscale'] = test_img_df['images'].map((lambda x: rgb2gray(x)))
test_img_df['Resized Grey'] = test_img_df['Greyscale'].map(lambda x: resize(x, (IMG_WIDTH, IMG_HEIGHT)))
test_img_df['Size'] = test_img_df['Greyscale'].map(lambda x: x.shape)
test_img_df['Class'] =test_img_df['images'].map(lambda x: "Light" if (np.mean(x[:,:,2]) + np.mean(x[:,:,0]) > 0.8) else "Dark")


# In[ ]:


results = []
sizes = test_img_df['Size']

for index, image in test_img_df.iterrows():
    testIm = image['Resized Grey']
    results.append(testIm.tolist())    
results =  np.asarray(results)
preds = model.predict(results)


# In[ ]:


resultsSized = []
#Resize the images back to original
for i in range(0,len(preds)):
    resultsSized.append(resize(preds[i], sizes[i]))


# In[ ]:


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    count = 0
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    for i in range(0, len(run_lengths)/2):
        count += run_lengths[2*i+1]
    if count > 20:
        return run_lengths

def prob_to_rles(x):
    lab_img = label(x > 0.5)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

test_ids = test_img_df['ImageId']
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(resultsSized[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len([x for x in rle if x is not None]))


# In[ ]:


sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
trythis = [x for x in rles if x is not None]
sub['EncodedPixels'] = pd.Series(trythis).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('MLPdataAugmentation.csv', index=False)

