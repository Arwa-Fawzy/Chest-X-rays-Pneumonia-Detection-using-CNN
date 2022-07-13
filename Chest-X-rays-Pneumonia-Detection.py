#Required libraries 

import tensorflow as tf
import os
import glob
import shutil
import cv2
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import matplotlib.cm as cmm
import imgaug.augmenters as iaa
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import random
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, auc, roc_curve
from keras import backend as k


#Choosing color palettes by seaborn to be suitable with the human eye detection
color = sns.color_palette()

#allows dynamic allocation of GPU memory
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40

#allowing dynamic growth during the allocation/preprocessing in the memory
config.gpu_options.allow_growth = True


# If PYTHONHASHSEED is set to an integer value, 
#it is used as a fixed seed for generating the hash randomization
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(111)

# multi-threading 
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

#setting the random seed in tensorflow at graph level
tf.random.set_seed(111)
# defining another session with above session configs through tensorflow
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# Setting the session in keras and making the whole augmentation  
aug.seed(111)

print(os.listdir("E:\CS\Robotics club\data\chest_xray"))
# it depends on your file's path 
data_dir = Path('E:\CS\Robotics club\data\chest_xray')
# the train directory
train_dir = data_dir / 'train'
# The validation directory
val_dir = data_dir / 'val'
# the test directory
test_dir = data_dir / 'test'


# classifying the training data into two sub-categories: normal and pneumonia 
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'


# preparing all images .. it ends with (.jpeg) extension 
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# creating an empty list for the training set 
#appending the data into this list within img_path and label format
train_data = []
train_labels = []

for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    train_data.append(img)
    train_labels.append(label)

for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=2)
    train_data.append(img)
    train_labels.append(label)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

print(np.shape(train_data))
print(np.shape(train_labels))

#classifying the test data into two sub-categories: normal and pneumonia 
normal_cases_dir = test_dir / 'NORMAL'
pneumonia_cases_dir = test_dir / 'PNEUMONIA'

# preparing all images .. it ends with (.jpeg) extension 
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# creating an empty list for the test set 
#appending the data into this list within img_path and label format

test_data = []
test_labels = []
test_img =[]

for img in normal_cases:
    test_img.append(img)
    img = cv2.imread(str(img))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    test_data.append(img)
    test_labels.append(label)

for img in pneumonia_cases:
    test_img.append(img)
    img = cv2.imread(str(img))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=2)
    test_data.append(img)
    test_labels.append(label)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("Total number of test examples: ", test_data.shape)
print("Total number of labels:", test_labels.shape)
normal = 0
innormal = 0
for i,x in test_labels:
    if x == 1.0:
        innormal += 1
    if x == 0.0:
        normal += 1
print(normal, innormal)
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    img = cv2.imread(str(test_img[i]))
    img = cv2.resize(img, (200, 200))
    ax.imshow(img)
plt.tight_layout()
plt.show()

normal_cases_dir = val_dir / 'NORMAL'
pneumonia_cases_dir = val_dir / 'PNEUMONIA'

normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

val_data = []
val_labels = []
val_img = []
for img in normal_cases:
    val_img.append(img)
    img = cv2.imread(str(img))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    val_data.append(img)
    val_labels.append(label)

for img in pneumonia_cases:
    val_img.append(img)
    img = cv2.imread(str(img))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=2)
    val_data.append(img)
    val_labels.append(label)

val_data = np.array(val_data)
val_labels = np.array(val_labels)

print("Total number of test examples: ", val_data.shape)
print("Total number of labels:", val_labels.shape)
normal = 0
innormal = 0
for i,x in val_labels:
    if x == 1.0:
        innormal += 1
    if x == 0.0:
        normal += 1
print(normal, innormal)

#convolution block 
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
model.add(SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.output_shape)
print(model.summary())

#adam can handle sparse gradients on noisy problems
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', tf.keras.metrics.AUC()])

# Fitting the model
history = model.fit(train_data, train_labels, batch_size=20, epochs=27, verbose=1, validation_data=(val_data, val_labels))
loss_train = history.history['auc']
loss_val = history.history['val_auc']
# traversing on the data 27 times
epochs = range(1,28)

#creating a plot for the loss accuracy 
plt.plot(epochs, loss_train, 'g', label='Training AUC')
plt.plot(epochs, loss_val, 'b', label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,28)
plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
plt.plot(epochs, loss_val, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluating the model for train 
train_loss, train_score, train_auc = model.evaluate(train_data, train_labels, batch_size=20)
print("Loss on train set: ", train_loss)
print("Accuracy on train set: ", train_score)
print("AUC on train set: ", train_auc)

# Evaluating the model for test 
test_loss, test_score, test_auc = model.evaluate(test_data, test_labels, batch_size=20)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)
print("AUC on test set: ", test_auc)


# knowing the predictions
preds = model.predict(test_data, batch_size=20)
preds = np.argmax(preds, axis=-1)

# Original labels
orig_test_labels = np.argmax(test_labels, axis=-1)

print(orig_test_labels.shape)
print(preds.shape)

# creating the confusion matrix plot
cm  = confusion_matrix(orig_test_labels, preds)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

# Calculate Precision and Recall
#confusion matrix is table or classifier for the positive and negative probabilties or results 
#recall = true positive / (true positive +false negative)
tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
#using the format syntax to peint the recall and precision inside the string 
print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))

# illustration by the plot 
plt.figure(figsize=(12,12))
truc_x =['Train','Test']
truc_y=[train_score,test_score]
colors_list = ['Red', 'Blue']
plt.bar(truc_x,truc_y, color = colors_list)
for index, value in enumerate(truc_y):
    plt.text(value, index,
             str(value))
plt.title('Training and Testing Accuracy')
plt.xlabel('Dataset')
plt.ylabel('Percent')
plt.show()
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    #transforming the dimension of the array into a batch
    #of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # creating a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # computing the gradient of the top predicted class for the input image
    # with respect to the activations of the last convolution layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    #the gradient of the output neuron 
    # with the output feature map of the last convolution layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # creating vector, and each entry is the intensity of the gradient
    # among a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiplying each channel in the feature map array
    # with respect to the importance of the top predicted class
    # then, suming all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, this step is for normalizing the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.5):
    # Loading the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    # Rescaling heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Using jet colormap to colorize heatmap
    jet = cmm.get_cmap("jet")

    # Using RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Creating an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimposing the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Saving the superimposed image
    superimposed_img.save(cam_path)


    return cam_path


last_conv_layer_name = "separable_conv2d_7"
img_size = (28,28)

# Removing last layer's softmax
model.layers[-1].activation = None

for i,k in enumerate(test_img[0:100]):
    plt.figure(figsize=(20,5))
    img_path = test_img[i]
    img = cv2.imread(str(test_img[i]))
    img = cv2.resize(img,(228,228))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title(f'ORIGINAL: label = {test_labels[i][0]}',size=14)
    
    img_array = np.expand_dims(test_data[i], axis=0)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name,pred_index=1)
    cam_path = save_and_display_gradcam(img_path, heatmap)
    img = cv2.imread(cam_path)
    img = cv2.resize(img, (228, 228))
    plt.subplot(1,3,2)
    plt.imshow(img)
    plt.title(f'Grad-CAM: Predict = {preds[i]}',size=14)
    
    plt.subplot(1,3,3)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(test_data[i].astype('double'), model.predict, top_labels=2, hide_color=0, num_samples=2)
    temp, mask = explanation.get_image_and_mask(1, positive_only=True, negative_only=False, num_features=5, hide_rest=True, min_weight=0.0)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title(f'LIME: Predict = {preds[i]}',size=14)
    
    
    plt.show()
    
