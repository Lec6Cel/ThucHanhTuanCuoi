import os
import urllib.request

# Kiểm tra xem tệp haarcascade_fullbody.xml đã tồn tại hay chưa
if not os.path.isfile('haarcascade_fullbody.xml'):
    # Nếu không tồn tại, tải tệp về từ URL
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml'
    urllib.request.urlretrieve(url, 'haarcascade_fullbody.xml')


import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load pre-trained pedestrian detection classifier
pedestrian_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Function to extract pedestrian bounding boxes from an image
def detect_pedestrians(image):
    if image is None:
        return []
    
    # Chuyển đổi hình ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện người đi bộ
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 4)
    
    return pedestrians


# Function to evaluate performance metrics
def evaluate_performance(true_labels, predicted_labels):
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return confusion_mat, accuracy, precision, recall, f1

# Load your dataset containing pedestrian and negative images
# Replace 'images_path' with the path to your image dataset
images_path = 'C:\\Users\\LENOVO\\OneDrive\\Máy tính\\computervision\\ThucHanhCuoi\\negative'

# labels: 1 for pedestrian images, 0 for negative images
labels = np.array([1, 0, 1, 0, 1, 0, ...])
predicted_labels = []

# Perform pedestrian detection on each image in the dataset
for i in range(len(images_path)):
    image = cv2.imread(images_path[i])
    pedestrians = detect_pedestrians(image)
    # If pedestrians are detected, draw rectangles around them
    if len(pedestrians) > 0:
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# Evaluate performance metrics
confusion_mat, accuracy, precision, recall, f1 = evaluate_performance(labels, predicted_labels)

# Print performance metrics
print("Confusion Matrix:")
print(confusion_mat)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Test with some images and draw rectangles around pedestrians
test_image_paths = ['path/to/test/image1.jpg', 'path/to/test/image2.jpg', ...]
for path in test_image_paths:
    test_image = cv2.imread(path)
    pedestrians = detect_pedestrians(test_image)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Pedestrian Detection', test_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
