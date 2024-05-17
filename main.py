import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('Model.h5')

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame for the model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (164, 1))  # Resize to match model input size
    img = img.flatten()  # Flatten the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=1)  # Add sequence length dimension

    # Normalize pixel values
    img = img / 255.0

    class_labels = [f"{i}" for i in range(1710)]

    # Make a prediction with the model
    prediction = model.predict(img)

    # Get the predicted class index
    class_idx = np.argmax(prediction)

    # Get the predicted class label
    class_label = class_labels[class_idx]

    # Display the frame and the predicted label
    cv2.putText(frame, f"Prediction: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    # Wait for 100 milliseconds (0.1 seconds)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()