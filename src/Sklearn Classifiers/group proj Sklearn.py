
# coding: utf-8

# In[41]:


import os
import sys

import numpy as np
import pandas as pd
import os
import time

# In[42]:




from PIL import Image

ALL_TRAIN_IMAGES = []
ALL_TRAIN_LABELS = []

location = "Data";
image_preprocessing_Start = time.time()

for dir in os.listdir(location):
    imageDetail = pd.read_csv(location + "/"+dir+"/GT-"+dir+".csv", sep=";")
    for i in range(0,(len(imageDetail))):
        #We perform all preprocessing on this step       
        
        img4 = Image.open(location + "/"+dir+"/"+imageDetail['Filename'][i])
        img5 = img4.crop((imageDetail['Roi.X1'][i],imageDetail['Roi.Y1'][i],imageDetail['Roi.X2'][i],imageDetail['Roi.Y2'][i]))

        image_object = img5.resize((50,50),Image.ANTIALIAS)
        imageArray = np.array(image_object)
        ALL_TRAIN_IMAGES.append(imageArray)
        ALL_TRAIN_LABELS.append(dir)
            
print 'Time for Preprocessing= %s SECONDS' %(time.time() - image_preprocessing_Start) 


# In[45]:


ALL_TRAIN_LABELS_NP = np.asarray(ALL_TRAIN_LABELS)
ALL_TRAIN_IMAGES_NP = np.asarray(ALL_TRAIN_IMAGES)

n_samples = len(ALL_TRAIN_IMAGES_NP)
data = ALL_TRAIN_IMAGES_NP.reshape((n_samples, -1))
labels = ALL_TRAIN_LABELS_NP
print("Total samples: " + str(len(ALL_TRAIN_LABELS_NP)))


# In[46]:


start_time = time.time()

import multiprocessing
cores = 1
if multiprocessing.cpu_count() > 4:
    cores = multiprocessing.cpu_count()-2

    
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler


#We dont perform normalization and Feature reduction here. We try to do it using Scikit-image


# 1.StandardScalar
# scaler = StandardScaler()
# scaler.fit(All_TRAIN_IMAGES_NP)
# scaler.transform(All_TRAIN_IMAGES_NP)

# 2.Normalize
# All_TRAIN_IMAGES_NP = normalize(ALL_TRAIN_IMAGES_NP)


#Feature Reduction
# 1. PCA
#pca=PCA(3000)
#data_pca_transformed = pca.fit_transform(data)
#data_pca_transformed = pca.fit_transform(normalized_data)
#print data_pca_transformed.shape
#print 'PCA variance ratio' + str(pca.explained_variance_ratio_.sum())

# 2. TruncatedSVD
#svd = TruncatedSVD(1500)
#data_svd_transformed = svd.fit_transform(data)
#data_svd_transformed = svd.fit_transform(normalized_data)
#print 'SVD variance ratio' + str(svd.explained_variance_ratio_.sum())


from sklearn.model_selection import train_test_split

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=0.80, random_state=42)





# In[ ]:


# 1. Linear SVC
from sklearn.svm import LinearSVC

print 'Starting Linear SVC'
start_time= time.time()
clf = LinearSVC()
clf.fit(data_train, labels_train)
print 'Score of SVC classification:' + clf.score(data_test, labels_test)
print 'Time Taken by Classifier = '+ str(time.time() - start_time) + 'seconds' 

import pickle
pickle.dump(clf, open("Pickle/SevenClasses/TrainedModels/svcClassifier.model", 'wb'))


# f1_score(y_true, y_pred, average='macro')  
# f1_score(y_true, y_pred, average='micro')  
# f1_score(y_true, y_pred, average='weighted')  
# f1_score(y_true, y_pred, average=None)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 2. Random Forest Classifier

print 'Starting Random Forest Classifier'
start_time= time.time()
clf = RandomForestClassifier(min_samples_leaf=100, n_jobs=cores)
clf.fit(data_train, labels_train)
scores = cross_val_score(clf, data_train, labels_train, cv=5)

print 'Score of RF classification:' + clf.score(data_test, labels_test)
print '5 fold Scores of RF classification:' + scores
print 'Time Taken by Classifier with 5 folds = '+ str(time.time() - start_time) + 'seconds' 

import pickle
pickle.dump(clf, open("Pickle/SevenClasses/TrainedModels/RandomForestClassifier.model", 'wb'))


# In[ ]:


from sklearn.neural_network import MLPClassifier


# 3. MLP Classifier

print 'Starting MLP Classifier'
start_time= time.time()

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,idden_layer_sizes=(100, 100),random_state=1)
clf.fit(data_train, labels_train)
print 'Score of RF classification:' + clf.score(data_test, labels_test)
print 'Time Taken by Classifier = '+ str(time.time() - start_time) + 'seconds' 

import pickle
pickle.dump(clf, open("Pickle/SevenClasses/TrainedModels/MLPClassifier.model", 'wb'))


# In[ ]:


# Load models from generated Pickles

import pickle
from sklearn.metrics import confusion_matrix

pickledLocation= "Pickle/AllClasses/TrainedModels/"
for eachFile in os.listdir(pickledLocation): 
    classifier = pickle.load( open( "Pickle/AllClasses/TrainedModels/"+ eachFile, "rb" ))
    labels_pred = classifier.predict(data_test)
    cm = confusion_matrix(labels_test, labels_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.log(.0001 + cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title(eachFile + 'log normalized Confusion Matrix')
    plt.show()
    



# In[ ]:


### Adding interactive UI to python Notebook, Not Fully Functional. For UI run webApp.

# from IPython.display import display
# from selectfile import FileBrowser
# import selectfile
# from Tkinter import Tk
# import tkFileDialog as filedialog

# button = widgets.Button(description="Click Me!")
# bounded = widgets.Text()


# def on_button_clicked(sender):
#     print(bounded.value)


# button.on_click(on_button_clicked)


# display(button)
# display(bounded)

