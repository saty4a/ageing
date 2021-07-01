#!/usr/bin/env python
# coding: utf-8

# # Import the necessary packages

# In[1]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths


# # Initialize the number of epochs to train for, and batch size

# In[2]:


EPOCHS = 10
BS = 32

DIRECTORY = r"dataset/Puffy_Eyes_Dataset"
CATEGORIES = ["Normal_Eyes", "Puffy_Eyes"]

print("loading images......")


# #  Grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and  labels

# In[3]:


data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(120, 120))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)


# # Perform one-hot encoding on the labels

# In[4]:


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)


# # Splitting the data into training and testing dataset

# In[5]:


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


# # Construct the training image generator for data augmentation

# In[6]:


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# # Load the EfficientNetB0 network

# In[7]:


baseModel = EfficientNetB0(weights="imagenet", include_top=False , input_shape=(120,120,3))


# # Step1: Specify the architecture

# In[8]:


model = Sequential() 
model.add(baseModel)
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(2,activation="softmax"))


# In[9]:


for layer in baseModel.layers:
     layer.trainable = False


# # Step2: Compile the model

# In[10]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# #  Step3: Train the model

# In[11]:


H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


# # Step4: Make predictions on the testing set

# In[12]:


print("[INFO] evaluating network...")
pred = model.predict(testX, batch_size=BS)


# In[13]:


pred[0]


# In[14]:


np.argmax(pred[0])


# # Serialize the model to disk

# In[15]:


print("saving puffy eye detector model.....")
model.save("puffyEye_detector.model",save_format="h5")


# # Plot the training loss and accuracy

# In[16]:


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("puffyEye.png")


# In[ ]:





# %%
