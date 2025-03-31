import os
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.dataset_loader import load_srd_dataset

def build_generator():
    inputs = Input(shape=(256, 256, 3))
    
    # Encoder
    e1 = Conv2D(64, 4, strides=2, padding='same')(inputs)
    e1 = LeakyReLU(0.2)(e1)
    
    e2 = Conv2D(128, 4, strides=2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)
    
    e3 = Conv2D(256, 4, strides=2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)

    # Bottleneck
    b = Conv2D(512, 4, strides=2, padding='same')(e3)
    b = BatchNormalization()(b)
    b = LeakyReLU(0.2)(b)

    # Decoder
    d1 = Conv2DTranspose(256, 4, strides=2, padding='same')(b)
    d1 = Concatenate()([d1, e3])
    d1 = LeakyReLU(0.2)(d1)
    
    d2 = Conv2DTranspose(128, 4, strides=2, padding='same')(d1)
    d2 = Concatenate()([d2, e2])
    d2 = LeakyReLU(0.2)(d2)
    
    d3 = Conv2DTranspose(64, 4, strides=2, padding='same')(d2)
    d3 = Concatenate()([d3, e1])
    d3 = LeakyReLU(0.2)(d3)

    outputs = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(d3)
    
    return Model(inputs, outputs)

def train():
    shadow_imgs, shadow_free_imgs = load_srd_dataset()
    
    generator = build_generator()
    generator.compile(loss='mae', optimizer=Adam(0.0002))
    
    # Train with paired data
    generator.fit(shadow_imgs, shadow_free_imgs,
                 batch_size=4,
                 epochs=50,
                 validation_split=0.1)
    
    generator.save('models/shadow_removal_model.h5')

if __name__ == '__main__':
    train()