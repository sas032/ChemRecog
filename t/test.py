#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:32:32 2021

@author: codiman
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import os




train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)


train_data =train.flow_from_directory('basedata/train/', target_size=(200,200),batch_size=3,class_mode='binary')