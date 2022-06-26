import os
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAINING_DIR = "./dataset/train/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    #rotation_range=40,
      validation_split=0.1,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True,
      vertical_flip=True)

VALIDATION_DIR = "./dataset/test/"
validation_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.1)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(300,300),
	class_mode='categorical',
  batch_size=32
  )

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(300,300),
	class_mode='categorical',
  batch_size=32
)
labels = (train_generator.class_indices)
labels = dict((v,k)for k,v in labels.items())

model = keras.models.load_model("./waste_model.h5")

####################################################################

import matplotlib.pyplot as plt

# Evaluating the model on test data

filenames = validation_generator.filenames
nb_samples = len(filenames)

model.evaluate_generator(validation_generator, nb_samples)

# Generating predictions on test data

test_x, test_y = validation_generator.__getitem__(1)
preds = model.predict(test_x)

# Comparing predcitons with original labels

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])

# Confusion Matrix

y_pred = model.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Accuracy score

acc = accuracy_score(validation_generator.classes, y_pred)
print("Accuracy is {} percent".format(round(acc*100,2)))