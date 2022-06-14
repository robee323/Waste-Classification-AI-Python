import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cardboard_dir = os.path.join('./dataset/train/cardboard')
glass_dir = os.path.join('./dataset/train/glass')
metal_dir = os.path.join('./dataset/train/metal')

print('total training cardboard images:', len(os.listdir(cardboard_dir)))
print('total training glass images:', len(os.listdir(glass_dir)))
print('total training metal images:', len(os.listdir(metal_dir)))

cardboard_files = os.listdir(cardboard_dir)
print(cardboard_files[:10])

glass_files = os.listdir(glass_dir)
print(glass_files[:10])

metal_files = os.listdir(metal_dir)
print(metal_files[:10])
###################################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(cardboard_dir, fname) 
                for fname in cardboard_files[pic_index-2:pic_index]]
next_paper = [os.path.join(glass_dir, fname) 
                for fname in glass_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(metal_dir, fname) 
                for fname in metal_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

###################################################################################################

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

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 64 neuron hidden layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
    ])
filepath="waste_model.h5"
checkpoint1=ModelCheckpoint(filepath,monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list=[checkpoint1]

model.summary()

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])

history = model.fit(train_generator, epochs=100, validation_data = validation_generator, verbose = 1, callbacks=callbacks_list)

file="waste_model.h5"
keras.models.save_model(model,file)

####################################################################

import matplotlib.pyplot as plt

# Training loss graph drawing
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#Accuracy graph drawing
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

# Accuracy score

acc = accuracy_score(validation_generator.classes, y_pred)
print("Accuracy is {} percent".format(round(acc*100,2)))