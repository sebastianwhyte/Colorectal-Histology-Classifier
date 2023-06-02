#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An image classification model that classifies colorectal histology images using Tensorflow

Created on Tue May 30 12:36:39 2023

@author: sebastian2
"""


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers


import numpy as np
import matplotlib.pyplot as plt



def configure_for_performance(ds, buffer_size: int, batch_size: int):
    """
    Configures the dataset for training -> Shuffle, batch, and have batches be availble asap.

    Parameters
    ----------
    ds : TYPE
        dataset to configure
    buffer_size : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.

    Returns
    -------
    ds : TYPE
       

    """
    
    ds = ds.cache()
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds
    


def scale(image, resize_and_rescale):
    """
    Resizes and scales an image.

    Parameters
    ----------
    image : TYPE
        image to rescale

    Returns
    -------
    None.

    """
    
    #print(image)
    
    result = resize_and_rescale(image)
    

    # Verify that the pixels are in the [0, 1] range.
    print('Min and max pixel values:', result.numpy().min(), result.numpy().max())

    return result
 
    


def preprocess_layers(image, data_augmentation):
    """
    Preprocesses layers and applies them repeatedly to the same image.
    
    RandomFlip randomly flips images during training.
    RandomRotation randomly rotates images during training. 

    Parameters
    ----------
    image : TYPE
        image to process

    Returns
    -------
    None.

    """
    
    
    # Add image to a batch
    image = tf.cast(tf.expand_dims(image, 0), tf.float32)
    
    plt.figure(figsize=(10,10))
    
    for i in range(9):
        augumented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augumented_image[0])
        plt.axis('off')
    



def prepare(ds, shuffle=False, augment=False):
    """
    Prepares the model for training. 

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    shuffle : TYPE, optional
        DESCRIPTION. The default is False.
    augment : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=AUTOTUNE)
    
    
    if shuffle:
        ds = shuffle(1000)
        
    # Batch all datasets.
    ds = ds.batch(BATCH_SIZE)
    
    
    

def get_images_and_labels(train_ds, metadata):
    """
    Retrieves the images and class names/labels from the training dataset.

    Parameters
    ----------
    train_ds : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.

    Returns
    -------
    images : TYPE
        DESCRIPTION.
    class_names : TYPE
        DESCRIPTION.

    """
   
    # Retrieve an image from the dataset.
    get_label_name = metadata.features['label'].int2str
    
    images = []
    class_names = []
    
    
    # Get the images and labels from the train dataset.
    for image, label in train_ds:
        images.append(image)
        class_names.append(get_label_name(label))
            
    return images, class_names
    
    
 
    
def show_images(images, class_names) -> None:
    """
    Displays the first 25 images from the TRAINING
    SET and displays the class name below each image.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(10,10))

    # Use subplot to display multiple images.
    for i in range(25):
        plt.subplot(5,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[i])
    
    plt.show()  
        
    
        
   
# Program entry point
if __name__ == '__main__':
    
    # Constants
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE
    
    plt.ion()
    
    
    # Create train, validation, and test sets by loaing the dataset.
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'colorectal_histology',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    
    
    # Get images and class names/labels.
    images, class_names = get_images_and_labels(train_ds, metadata)  
    
    # Get the number of classes from the metadata features.
    num_classes = metadata.features['label'].num_classes
    
    print(num_classes)
    
    # DEBUG
    #for i in class_names:
    #    print(i)
    
    
    
    # Visualize the data.   
    show_images(images, class_names) 
    plt.imshow(images[0])
    plt.show()
       
    
    IMG_SIZE = 180
    
    # Create layers for resizing and rescaling the image.
    resize_and_rescale = keras.Sequential([
        layers.Resizing(height=IMG_SIZE, width=IMG_SIZE),
        layers.Rescaling(scale=1./255)
        ])
    
    
    
    # Visualize the data
    #Note: Image will be enlarged because original size was 150 x 150 pixels.
    result = plt.imshow(scale(images[0], resize_and_rescale)) 
    plt.show()
    
    
    
    # Retrieve the first image. Resize and rescale it.
    result = scale(images[0], resize_and_rescale)
    
    
    # Create preprocessing layers.
    data_augmentation = keras.Sequential([
        layers.RandomFlip(mode='horizontal_and_vertical'),
        layers.RandomRotation(factor=0.2)
        ])
    
    
    # Preprocess the image.
    preprocess_layers(result, data_augmentation)
    

    # Create a standard model -- not tuned for accuracy yet.
    model = keras.Sequential([
        # Add preprocessing layers first
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
    
    
    # Compile the model.
    
    
    
    
    
    '''
    
    # Batch, shuffle, and configure the training, validation, and test sets for preformance.
    train_ds = configure_for_performance(train_ds, BUFFER_SIZE, BATCH_SIZE)
    val_ds = configure_for_performance(val_ds, BUFFER_SIZE, BATCH_SIZE)
    test_ds = configure_for_performance(test_ds, BUFFER_SIZE, BATCH_SIZE)
    
    '''