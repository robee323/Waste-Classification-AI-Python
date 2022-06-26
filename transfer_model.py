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
training_datagen = ImageDataGenerator(rescale = 1./255)

VALIDATION_DIR = "./dataset/test/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(300,300),
  color_mode="rgb",
	class_mode='categorical',
  batch_size=32,
  shuffle=True
  )

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(300,300),
  color_mode="rgb",
	class_mode='categorical',
  batch_size=32,
  shuffle=True
)
labels = (train_generator.class_indices)
labels = dict((v,k)for k,v in labels.items())

# instantiate a base model with pre-trained weights
base_model = tf.keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(300, 300, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.
# Not train first layer of model. It is already trained
base_model.trainable = False

#a new model on top
inputs = keras.Input(shape=(300, 300, 3))
# We make sure that the base_model is running in inference mode here
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# A Dense classifier with a six category classification
outputs = keras.layers.Dense(6,  activation='softmax')(x)
model = keras.Model(inputs, outputs)


model.summary()

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])

history = model.fit(train_generator, validation_data=validation_generator,verbose = 1, epochs=50)


file="waste_transfer_model.h5"
keras.models.save_model(model,file)

##########################################################

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['categorical_accuracy'], label='acc')
plt.plot(history.history['val_categorical_accuracy'], label='val_acc')
plt.legend()
plt.show()

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

# Accuracy

acc = accuracy_score(validation_generator.classes, y_pred)
print("Accuracy is {} percent".format(round(acc*100,2)))
