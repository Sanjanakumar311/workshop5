# Workshop5
## License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier 
## PROGRAM 
```py
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

image_path = 'my image.jpg'  # <-- Change this to your image filename
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the 'image_path' variable.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Histogram Equalization for better contrast
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image (Blur + Equalized)")
plt.axis('off')
plt.show()

cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    print("Cascade file not found. Downloading...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
    print("Cascade file downloaded successfully!")

face_cascade = cv2.CascadeClassifier(cascade_path)

faces = face_cascade.detectMultiScale(
    equalized,          # Preprocessed grayscale image
    scaleFactor=1.1,    # Scaling factor between image pyramid layers
    minNeighbors=5,     # Higher value -> fewer false detections
    minSize=(30, 30)    # Minimum object size
)

print(f"Total Faces Detected: {len(faces)}")
output = image.copy()
save_dir = "Detected_Faces"
os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_crop = image[y:y+h, x:x+w]
    save_path = f"{save_dir}/face_{i+1}.jpg"
    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("⚠️ No faces detected. Try adjusting parameters or using a clearer image.")

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()
```
## OUTPUT: 
Original image:

<img width="235" height="412" alt="image" src="https://github.com/user-attachments/assets/6a91346d-a30c-404e-af78-ac183cd45e62" />

Grayscale image:

<img width="238" height="412" alt="image" src="https://github.com/user-attachments/assets/abd0c6df-fcc4-4e70-a1e9-62a08e7ed926" />

Preprocessed image:

<img width="335" height="417" alt="image" src="https://github.com/user-attachments/assets/759c11ba-81e9-4114-b901-520a3268fa60" />

Detected faces:

<img width="255" height="418" alt="image" src="https://github.com/user-attachments/assets/3d55e3eb-ce81-4fca-98ed-cfe941fa8efb" />

## Result:
Thus, the program was executed successfully.
