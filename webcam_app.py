""" webcam_app.py """

import cv2
import torch
from PIL import Image
from fer_model_loader import FERModelLoader

if __name__ == "__main__":
    # Load facial emotion recognition model
    model_loader = FERModelLoader(weights_file_path="fer_resnet18.pth")
    fer_model = model_loader.get_model()
    transformation = model_loader.get_transformation(transformation_file="fer_transformation.pth")

    # Load face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Start webcam, try opening the first camera (index 0)
    webcam = cv2.VideoCapture(0)
        
    if not webcam.isOpened():
        print("Error: Could not access the webcam. Trying other devices...")
        
        # Try opening additional cameras
        for i in range(1, 5):
            webcam = cv2.VideoCapture(i)
            if webcam.isOpened():
                print(f"Camera {i} opened successfully.")
        else:
            print("Error: No webcam found. Please make sure a webcam is connected.")
    else:
        print("Camera 0 opened successfully.")

    while True:
        # Try to read a frame, and comfirm it was read with ret flag
        ret, frame = webcam.read()
        if not ret:
            break

        # Convert the frame to grayscale and scan frame for faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Iterate over each face and classify
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]

            # Convert the face image to PIL format
            face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            # Apply the transformation
            input_tensor = transformation(face_image_pil).unsqueeze(0)
            with torch.no_grad():
                output = fer_model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                label = model_loader.emotion_classes[prediction]

            # Create a rectangle arount the user(s) and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Facial Emotion Recognition', frame)

        # This allows the user to type q to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
