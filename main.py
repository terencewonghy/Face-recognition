import cv2
import os
import random

# Set the path to the FDDB dataset
fddb_path = 'D:/face_recognition/FDDB'

# Set the size of the positive samples
sample_size = (128, 128)

# Create a directory to store the positive samples
positive_samples_dir = 'D:/face_recognition/FDDB/positive_samples'
os.makedirs(positive_samples_dir, exist_ok=True)

# Initialize a counter for the positive samples
counter = 0

for i in range(1, 11):
    # Set the path to the annotations file
    annotations_path = os.path.join(fddb_path, 'FDDB-folds', f'FDDB-fold-{i:02d}-ellipseList.txt')

    # Create a directory for this fold's positive samples
    fold_positive_samples_dir = os.path.join(positive_samples_dir, f'fold_{i:02d}')
    os.makedirs(fold_positive_samples_dir, exist_ok=True)

    # Open the annotations file
    with open(annotations_path, 'r') as f:
        # Read the annotations line by line
        for line in f:
            # Get the image path and replace backslashes with forward slashes
            image_path = os.path.join(fddb_path, 'originalPics', line.strip().replace('\\', '/') + '.jpg')
            # Load the image
            image = cv2.imread(image_path)
            # Get the number of faces in the image
            num_faces = int(f.readline().strip())
            # Loop over the faces
            for i in range(num_faces):
                # Get the face annotation
                annotation = f.readline().strip().split()
                # Get the major and minor axis radii and angle of the ellipse
                ra, rb, theta = map(float, annotation[:3])
                # Get the center coordinates of the ellipse
                center_x, center_y = map(float, annotation[3:5])
                # Convert ellipse to a rotated rectangle
                box = cv2.boxPoints(((center_x, center_y), (ra * 2, rb * 2), theta))
                box = box.astype(int)
                # Get the bounding rectangle of the rotated rectangle
                x, y, w, h = cv2.boundingRect(box)
                # Check if the bounding rectangle is within the image boundaries
                if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                    # Crop the face from the image
                    face = image[y:y + h, x:x + w]
                    # Resize the face to the sample size
                    face = cv2.resize(face, sample_size, interpolation=cv2.INTER_CUBIC)
                    # Save the positive sample in this fold's directory
                    sample_path = os.path.join(fold_positive_samples_dir, f'face_{counter}.jpg')
                    cv2.imwrite(sample_path, face)
                    # Increment the counter
                    counter += 1

# Print total number of positive samples generated.
print(f'Total number of positive samples generated: {counter}')

###########################################################################################
# Set the path to the FDDB dataset
fddb_path = 'D:/face_recognition/FDDB'

# Set the path to the FDDB-folds directory
fddb_folds_path = os.path.join(fddb_path, 'originalPics')

# Create a list to store the paths to the non-face images
non_face_images = []

# Recursively search for image files within the FDDB-folds directory
for root, dirs, files in os.walk(fddb_folds_path):
    for file in files:
        # Check if the file is an image file
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # Get the full path to the image file
            image_path = os.path.join(root, file)
            # Add the image path to the list of non-face images
            non_face_images.append(image_path)

# Set the patch size
patch_width = 128
patch_height = 128

# Set the number of patches to crop from each image
num_patches = 10

# Create a directory to store the negative samples
negative_samples_dir = 'D:/face_recognition/FDDB/negative_samples'
os.makedirs(negative_samples_dir, exist_ok=True)

# Load the Haar Cascade classifier for faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a counter for the negative samples
counter = 0

for image_path in non_face_images:
    # Load a non-face image
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    image_height, image_width, _ = image.shape

    for i in range(num_patches):
        # Generate random x and y coordinates for the top-left corner of the patch
        x = random.randint(0, image_width - patch_width)
        y = random.randint(0, image_height - patch_height)

        # Crop the patch from the image
        patch = image[y:y + patch_height, x:x + patch_width]

        # Convert the patch to grayscale
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Detect faces in the patch
        faces = face_cascade.detectMultiScale(gray_patch)

        # Only save the patch as a negative sample if no faces are detected
        if len(faces) == 0:
            # Save the negative sample with a unique filename
            sample_path = os.path.join(negative_samples_dir, f'negative_{counter}.jpg')
            print(f'Saving negative sample: {sample_path}')
            cv2.imwrite(sample_path, patch)

            # Increment the counter
            counter += 1

            # Break out of the loop once 10k negative samples have been generated
            if counter >= 10000:
                break

    # Break out of the outer loop once 10k negative samples have been generated
    if counter >= 10000:
        break
# Print total number of positive samples generated.
print(f'Total number of negative samples generated: {counter}')