
# coding: utf-8

# In[2]:


from skimage.morphology import label
import pandas as pd
from skimage.color import rgb2gray
import numpy as np
from glob import glob
from skimage.transform import resize
import os
import imageio
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K



# In[3]:


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


# In[14]:


#create training data and targets
X = []
t = []


for index, image in train_img_df.iterrows():
    tempIm = image["images"].tolist()
    tempMask = image['masks'].tolist()
    X.append(tempIm)
    t.append(tempMask)


# In[15]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))


s = Lambda(lambda x: x) (inputs)

#create model

c1 = Convolution2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Convolution2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Convolution2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Convolution2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Convolution2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Convolution2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Convolution2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Convolution2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Convolution2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Convolution2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Deconvolution2D(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 =  merge([u6, c4], mode='concat')
c6 = Convolution2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Convolution2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Deconvolution2D(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = merge([u7, c3], mode='concat')
c7 = Convolution2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Convolution2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Deconvolution2D(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = merge([u8, c2], mode='concat')
c8 = Convolution2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Convolution2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Deconvolution2D(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = merge([u9, c1], mode='concat',concat_axis=3)
c9 = Convolution2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Convolution2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Convolution2D(1, (1, 1), activation='sigmoid') (c9)


model = Model(input=[inputs], output=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.summary()

#fit model
model.fit(X, t,nb_epoch=30,batch_size=32,validation_split=0.1)

print "Model completed"


# In[16]:

#Load test data
test_img_df = image_df.query('TrainingSplit=="FullTesting"')
IMG_WIDTH = 256; IMG_HEIGHT = 256;
IMG_CHANNELS = 3

test_img_df['images'] = test_img_df['path'].map(lambda x: imageio.imread(str(x))[:,:,:IMG_CHANNELS]/float(255))
test_img_df['Greyscale'] = test_img_df['images'].map((lambda x: rgb2gray(x)))
test_img_df['Resized Grey'] = test_img_df['Greyscale'].map(lambda x: resize(x, (IMG_WIDTH, IMG_HEIGHT)))
test_img_df['Size'] = test_img_df['Greyscale'].map(lambda x: x.shape)
test_img_df['Class'] =test_img_df['images'].map(lambda x: "Light" if (np.mean(x[:,:,2]) + np.mean(x[:,:,0]) > 0.8) else "Dark")


# In[1]:


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
    #only return if cell area is more than 20 pixels
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
sub.to_csv('CNN.csv', index=False)

