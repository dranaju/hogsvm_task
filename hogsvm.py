import os
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_data_dir = '../archive/dogvscat_small/train'

def normalize_image(image):
    return image / 255.0

def load_data(data_dir, img_width=84, img_height=84):
    datas = []
    labels = []
    
    class_names = sorted(os.listdir(data_dir))
    #print('class_names:', class_names)

    for class_index, class_name in enumerate(class_names):
        label = class_index
        class_dir = os.path.join(data_dir, class_name)

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).convert('L')
            image = image.resize((img_width, img_height))
            image = normalize_image(np.array(image))

            datas.append(image)
            labels.append(label)


    return np.array(datas), np.array(labels)


train_datas, train_labels = load_data(train_data_dir)

X_train, X_test, y_train, y_test = train_test_split(train_datas, train_labels, test_size=0.3)

X_train_hog = []
X_train_hog_features = []
X_test_hog = []
X_test_hog_features = []
for image_train in X_train:
    fd, hog_image = hog(image_train, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    X_train_hog.append(hog_image)
    X_train_hog_features.append(fd)

for image_test in X_test:
    fd, hog_image = hog(image_test, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    X_test_hog.append(hog_image)
    X_test_hog_features.append(fd)

X_train_hog_features = np.array(X_train_hog_features)
X_test_hog_features = np.array(X_test_hog_features)

params = {'C': [1, 10, 100],
          'gamma': ['scale', 'auto', 0.1, 0.01],
          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

clf = svm.SVC()

grid = GridSearchCV(clf, params, refit=True, verbose=3, cv=4)

grid.fit(X_train_hog_features, y_train)

Y_pred_train = grid.predict(X_train_hog_features)
accuracy = accuracy_score(y_train, Y_pred_train)
print(f'Accuracy train: {100*accuracy:.2f}')

Y_pred = grid.predict(X_test_hog_features)
accuracy = accuracy_score(y_test, Y_pred)
print(f'Accuracy test: {100*accuracy:.2f}')

print(grid.best_params_)
