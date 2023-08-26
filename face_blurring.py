import cv2
import numpy as np

def detect_and_blur_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml') 
    # You can download the XML file from: https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply blur to faces
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
        image[y:y+h, x:x+w] = blurred_roi

    # Return the image with blur applied to faces
    return image

if __name__ == "__main__":
    # Input image file path
    image_path = input(r"Please enter the image path that contains the face:")

    # Detect and blur faces
    blurred_image = detect_and_blur_faces(image_path)

    # Save the resulting image
    cv2.imwrite("output_image.jpg", blurred_image)

    print("Face blurring completed.")