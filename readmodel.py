import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import seaborn
seaborn.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization,ReLU
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from sklearn.metrics import f1_score
import scipy.stats as stats
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# This is the kaggle dataset path
datapath=r'../input/pizza-images-with-topping-labels/pizza_data/labels.csv'
imgpath=r'../input/pizza-images-with-topping-labels/pizza_data/images'

# datapath=r'pizza_data/labels.csv'
# imgpath=r'pizza_data/images'
df=pd.read_csv(datapath)
df=df.sample(n=9000, replace=False, random_state = 1)
df['plain']=0
columns=df.columns

df['image_name']=df['image_name'].apply(lambda x: os.path.join(imgpath,x))
df = df.rename(columns={'image_name': 'filepaths'})

for i in range (len(df)):
    label_list=[]
    for j in range (1,len(df.columns)-1):
        column=df.columns[j]        
        label=df.iloc[i][column]
        label_list.append(label)
    max=np.max(label_list)        
    if max == 0:         
        df['plain'].iloc[i]=1
train_df, dummy_df=train_test_split(df, train_size=.8, shuffle=True, random_state=123)
valid_df, test_df=train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123)
print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df)) 

img_size=(300,300)
columns=df.columns[1:]
class_count=len(columns)
generator=ImageDataGenerator()
train=generator.flow_from_dataframe(train_df, x_col='filepaths', y_col=columns, target_size=img_size, batch_size=30,shuffle=True,seed=123, class_mode='raw')
val=generator.flow_from_dataframe(valid_df, x_col='filepaths', y_col=columns, target_size=img_size, batch_size=30,shuffle=False,class_mode='raw')
test=generator.flow_from_dataframe(valid_df, x_col='filepaths', y_col=columns, target_size=img_size, batch_size=30,shuffle=False,class_mode='raw')

def display_samples(img, classes): 
    images,labels=next(img)
    plt.figure(figsize=(25, 25))
    num_of_images=10
    for i in range(num_of_images):        
        plt.subplot(5, 5, i + 1)
        image=images[i] /255 
        label=labels[i]
        title=''
        for j in range(len(label)):
            value=label[j]
            if value == 1:
                title=title + ' '+ classes[j]
        plt.imshow(image)
        fname=os.path.basename(img.filenames[i])        
        plt.title(title, fontsize=15)
    plt.show()
    
display_samples(train, columns )


# ------------------ Main
def display_samples(img, classes): 
    images,labels=next(img)
    plt.figure(figsize=(25, 25))
    num_of_images=10
    for i in range(num_of_images):        
        plt.subplot(5, 5, i + 1)
        image=images[i] /255 
        label=labels[i]
        title=''
        for j in range(len(label)):
            value=label[j]
            if value == 1:
                title=title + ' '+ classes[j]
        plt.imshow(image)
        fname=os.path.basename(img.filenames[i])        
        plt.title(title, fontsize=15)
    plt.show()
    
display_samples(train, columns )
lr=.001
# model=VOneNet()
# model=VOneNet_with_four_GFBs()
# model=VOneNet_with_eight_GFBs()
model=VOneNet_with_Only_Simple_GFBs()
# model=EfficientNet_B0_Dense1024(img_size, lr, class_count)
epochs=50
history=model.fit(x=train,  epochs=epochs, verbose=1, validation_data=val, shuffle=True,  initial_epoch=0)

model.save_weights('/kaggle/working/ModelOutput')
# -------------

def tr_plot(tr_data, start_epoch):
    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(vloss)
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(25,10))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].scatter(Epochs, tloss, s=100, c='red')    
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs', fontsize=18)
    axes[0].set_ylabel('Loss', fontsize=18)
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].scatter(Epochs, tacc, s=100, c='red')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs', fontsize=18)
    axes[1].set_ylabel('Accuracy', fontsize=18)
    axes[1].legend()
    plt.tight_layout    
    plt.show()
    return index_loss
    
loss_index=tr_plot(history,0)

loss, accuracy=model.evaluate(test)
print (accuracy)