import cv2
import glob
import numpy as np
import os

# Set the paths to the positive and negative samples
pos_dir = r'D:\face_recognition\FDDB\positive_samples'
neg_path = r'D:\face_recognition\FDDB\negative_samples\*.jpg'

# Set the size of the samples
sample_height, sample_width = 224, 224

# Load and resize the positive samples
pos_imgs = []
for root, dirs, files in os.walk(pos_dir):
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (sample_width, sample_height))
            pos_imgs.append(img_resized)

# Load and resize the negative samples
neg_imgs = []
for f in glob.glob(neg_path):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (sample_width, sample_height))
    neg_imgs.append(img_resized)

# Create an array to store the samples and labels
samples = np.zeros((len(pos_imgs) + len(neg_imgs), sample_height * sample_width), dtype=np.float32)
labels = np.zeros(len(pos_imgs) + len(neg_imgs), dtype=np.int32)

# Add the positive samples and labels
for i, img in enumerate(pos_imgs):
    samples[i] = img.flatten()
    labels[i] = 1

# Add the negative samples and labels
for i, img in enumerate(neg_imgs):
    samples[len(pos_imgs) + i] = img.flatten()
    labels[len(pos_imgs) + i] = 0

# Save the samples and labels to a file
np.savez('samples.npz', samples=samples, labels=labels)