import cv2
import tensorflow as tf
import numpy as np
# Load the pre-trained Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Load the image and convert it to grayscale
image = cv2.imread('mum1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop over the detected faces
for (x, y, w, h) in faces:
    # Extract the region of interest from the image
    roi = gray[y:y + h, x:x + w]

    # Preprocess the ROI for your model
    roi_resized = cv2.resize(roi, (224, 224))
    roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
    roi_normalized = roi_resized / 255.0

    # Make a prediction on the ROI using your model
    prediction = model.predict(np.expand_dims(roi_normalized, axis=(0, -1)))

    # Check if the prediction is above a certain threshold
    if prediction[0, 0] > 0.5:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()