import os
import cv2
import numpy as np

def load_srd_dataset(dataset_path='data/SRD/train', img_size=(256, 256)):
    shadow_path = os.path.join(dataset_path, 'shadow')
    shadow_free_path = os.path.join(dataset_path, 'shadow_free')
    
    shadow_images = []
    shadow_free_images = []
    
    for img_name in os.listdir(shadow_path):
        base_name = os.path.splitext(img_name)[0]
        shadow_free_name = f"{base_name}_no_shadow.jpg"
        
        shadow_img = cv2.imread(os.path.join(shadow_path, img_name))
        shadow_free_img = cv2.imread(os.path.join(shadow_free_path, shadow_free_name))
        
        if shadow_img is not None and shadow_free_img is not None:
            # Resize and normalize
            shadow_img = cv2.resize(shadow_img, img_size) / 127.5 - 1
            shadow_free_img = cv2.resize(shadow_free_img, img_size) / 127.5 - 1
            
            shadow_images.append(shadow_img)
            shadow_free_images.append(shadow_free_img)
    
    return np.array(shadow_images), np.array(shadow_free_images)