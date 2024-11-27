import os
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

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
            image = Image.open(image_path).convert('RGB')
            image = image.resize((img_width, img_height))
            image = normalize_image(np.array(image))

            datas.append(image)
            labels.append(label)


    return np.array(datas), np.array(labels)


train_datas, train_labels = load_data(train_data_dir)

X_train, X_test, y_train, y_test = train_test_split(train_datas, train_labels, test_size=0.3, random_state=1)

print('X_train:', X_train.shape)
plt.imshow(X_train[0])
plt.show()
print('y_train:', y_train[0])

for i in range(10):
    print('y_train:', y_train[i])
    plt.imshow(X_train[i])
    plt.show()
