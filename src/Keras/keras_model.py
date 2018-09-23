import numpy
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, Flatten
from keras.layers import Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_yaml

import pickle
import numpy as np
import matplotlib.pyplot as plot
import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
from sklearn.metrics import confusion_matrix
    

# random seed
seed = 20

# Get the label class and sign name mapping from the file signnames.csv file
signmap = []
with open('/Users/poojithaamin/Downloads/Traffic_Signs_data/signnames.csv', 'r', newline='') as csvfile:
    signs = csv.reader(csvfile, delimiter=',')
    next(signs,None)
    for row in signs:
        signmap.append(row[1])
    csvfile.close()

#print the sign names
signmap    

#load the training, test and validation files  
training_file = "/Users/poojithaamin/Downloads/Traffic_Signs_data/train.p"
validation_file= "/Users/poojithaamin/Downloads/Traffic_Signs_data/valid.p"
testing_file = "/Users/poojithaamin/Downloads/Traffic_Signs_data/test.p"



#read the train, test and validation files
with open(training_file, mode='br') as f:
    train = pickle.load(f)
with open(validation_file, mode='br') as f:
    valid = pickle.load(f)
with open(testing_file, mode='br') as f:
    test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#get the counts
count_train = len(X_train)
count_test = len(X_test.shape)
count_validation = len(X_valid.shape)
count_classes = len(set(y_train))

print("Count of records in training dataset: ", count_train)
print("Count of records in testing dataset: ", count_test)
print("Count of records in validation dataset: ", count_validation)
print("Count of classes =", count_classes)

#Display images
def display(dataset, dataset_y, ylabel="", cmap=None):
    plot.figure(figsize=(15, 16))
    for i in range(6):
        plot.subplot(1, 5, i+1)
        indx = random.randint(0, len(dataset))
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plot.imshow(dataset[indx], cmap = cmap)
        plot.xlabel(signmap[dataset_y[indx]])
        plot.ylabel(ylabel)
    plot.show()
    
display(X_train, y_train, "Training example")
display(X_test, y_test, "Testing example")
display(X_valid, y_valid, "Validation example")

#Plot histogram
def hist_plt(dataset, label):     
    plot.style.use('seaborn')
    hist, bins = np.histogram(dataset, bins=43)
    width = 0.9 * (bins[1] - bins[0])
    ctr = (bins[:-1] + bins[1:]) / 2
    plot.bar(ctr, hist, align='center', width=width, color='blue')
    plot.xlabel(label)
    plot.ylabel("No. Of Images")     
    plot.show()
    
# Plot histograms for all the training set
hist_plt(y_train, "Training Set Class Labels")


#Shuffle the dataset
X_train, y_train = shuffle(X_train, y_train)

#Convert to gray scale
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


gray_images = list(map(convert_to_gray, X_train))
display(gray_images, y_train, "Gray Scale image", "gray")

#local histogram equilization
def convert_local_histogram(image):
    knl = morp.disk(30)
    img_local = rank.equalize(image, selem=knl)
    return img_local


equalized_images = list(map(convert_local_histogram, gray_images))
display(equalized_images, y_train, "Equalized Image", "gray")


#Normalize image data
def image_normalize(image):
    image = image/255
    return image


n_training = X_train.shape
normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
for i, img in enumerate(equalized_images):
    normalized_images[i] = image_normalize(img)
normalized_images = normalized_images[..., None]

#PrePorcess method
def preprocess(data):
    grayimgs = list(map(convert_to_gray, data))
    equalizedimgs = list(map(convert_local_histogram, grayimgs))
    n_training = data.shape
    normalizedimgs = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalizedimgs):
        normalizedimgs[i] = image_normalize(img)
    normalizedimgs = normalizedimgs[..., None]
    return normalizedimgs


# class values encoding for train
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y_train = encoder.transform(y_train)
act_y_train = np_utils.to_categorical(encoded_Y_train)

# class values encoding for test
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Y_test = encoder.transform(y_test)
act_y_test = np_utils.to_categorical(encoded_Y_test)

# class values encoding for validation
encoder = LabelEncoder()
encoder.fit(y_valid)
encoded_Y_valid = encoder.transform(y_valid)
act_y_valid = np_utils.to_categorical(encoded_Y_valid)


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(0.5)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(0.5)))
    
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(43, activation='softmax'))    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#Preprocess Validation dataset
X_valid_preprocessed = preprocess(X_valid)

#Build the model
estimator = KerasClassifier(build_fn=cnn_model, epochs=20, batch_size=10, verbose=1)

#Fit the model with training dataset
hist = estimator.fit(normalized_images, act_y_train, validation_data=(X_valid_preprocessed, act_y_valid))

#Preprocess Test dataset
X_test_preprocessed = preprocess(X_test)

#Predict on test dataset
predictions = estimator.predict(X_test_preprocessed)

#Calculate F1 score for test prediction
from sklearn.metrics import f1_score
f1_score(encoded_Y_test, predictions, average='micro')

#Print predictions
print(predictions)

#Build confusion matrix
cm = confusion_matrix(encoded_Y_test, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.log(.0001 + cm)
plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Blues)
plot.title('Log of normalized Confusion Matrix')
plot.ylabel('True label')
plot.xlabel('Predicted label')
plot.show()


# Test with new image
tst_img = []
path = '/Users/poojithaamin/Downloads/Traffic_Signs_data/test_new/'
for img in os.listdir(path):
    print(img)
    image = cv2.imread(path + img)
    print (image)
    image = cv2.resize(image, (32,32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tst_img.append(image)


tst_img_preprocessed = preprocess(np.asarray(tst_img))

predictions = estimator.predict(tst_img_preprocessed)
predictions[0]

# Loss plot
plot.figure(figsize=[10,7])
plot.plot(hist.history['loss'],'r',linewidth=5.0)
plot.plot(hist.history['val_loss'],'b',linewidth=5.0)
plot.legend(['Training loss', 'Validation Loss'],fontsize=18)
plot.xlabel('Epochs ',fontsize=16)
plot.ylabel('Loss',fontsize=16)
plot.title('Loss Curves',fontsize=16)
 
# Accuracy plot
plot.figure(figsize=[10,7])
plot.plot(hist.history['acc'],'r',linewidth=5.0)
plot.plot(hist.history['val_acc'],'b',linewidth=5.0)
plot.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plot.xlabel('Epochs ',fontsize=16)
plot.ylabel('Accuracy',fontsize=16)
plot.title('Accuracy Curves',fontsize=16)


#kfold cross validation
kfoldvalidation = KFold(n_splits=3, shuffle=True, random_state=seed)

kfold_results = cross_val_score(estimator, X_test_preprocessed, act_y_test, cv=kfoldvalidation)
print("Output: %.2f%% (%.2f%%)" % (kfold_results.mean()*100, kfold_results.std()*100))
kfold_results

'''
######################
#Pickle and load and test

filename = '/Users/poojithaamin/Downloads/Traffic_Signs_data/test_new/keras.p'
pickle.dump(estimator, open(filename, 'wb'))

keras_loaded_model = pickle.load(open(filename, 'rb'))
keras_loaded_model.predict(new_test_images_preprocessed)
'''