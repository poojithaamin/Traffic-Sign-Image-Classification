from flask import Flask, render_template ,request, jsonify
import cv2
import os
import numpy as np
import pickle
from PIL import Image
import sklearn
from sklearn.svm import LinearSVC
from numpy import genfromtxt
import pandas as pd



APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'images/')

app = Flask(__name__,template_folder='templates')

extension = "png"
fileName = "noimage.png"

classNames =[]

@app.route('/')
def main():
    return render_template('hello.html', name='Suraj')

def predict(classifierName="linSvcClassifier.p"):
    global fileName, extension,classNames

    classifierRoot = APP_ROOT + "\\classifiers\\" +classifierName

    classifier = pickle.load(open(classifierRoot, "rb"))

    print(type(classifier))

    img4 = Image.open( target + "\\" + "processed." + extension )

    image_object = img4.resize((50, 50), Image.ANTIALIAS)

    imageArray = np.array(image_object)
    listofArrays= []
    listofArrays.append(imageArray)
    PREDICT_IMAGES_NP = np.asarray(listofArrays)

    predict = PREDICT_IMAGES_NP.reshape((1, -1))
    classLabel = classifier.predict(predict)

    classNumber = int(classLabel)
    className = classNames[1][classNumber+1]

    return jsonify(
        imgClass=className
    )





def processImage(file):
    global fileName,extension
    bgr = [30, 30, 200]
    thresh = 40
    # bright = cv2.imread('C:/Users/Suraj/PycharmProjects/255/Bright.png')
    # dark = cv2.imread(file)

    bright = cv2.imread(target + "\\" + fileName) #TODO : remove hard coding

    bright2 = cv2.imread(target + "\\" + fileName)

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(bright, minBGR, maxBGR)
    resultBGR = cv2.bitwise_and(bright, bright, mask=maskBGR)
    ret, threshed_img = cv2.threshold(resultBGR,
                                      127, 255, cv2.THRESH_BINARY)


    grayImg = cv2.cvtColor(threshed_img, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(grayImg, 15, 75, 75)

    image, contours, hier = cv2.findContours(bilateral, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    maxW =0
    maxH =0
    boundX=0
    boundY=0


    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(bright, (x, y), (x + w, y + h), (125, 125, 125), 2)

        if (maxW <= w and maxH <= h):
            maxW = w
            maxH = h
            boundX = x
            boundY = y

    boundary = 5
    imgCropped = bright2[boundY - boundary:boundY + maxW + boundary, boundX - boundary:boundX + maxW + boundary]
    cv2.imwrite(target + "\\" + "processed."+extension, imgCropped)


@app.route('/upload_avatar',methods=['POST'])
def upload():
    global fileName, extension, classNames

    #initialize Class names
    if len(classNames)==0:
        classNames = pd.read_csv('signnames.csv', sep=',', header=None)

    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files.get('file')
    fileName=file.filename
    extension= fileName.split('.')[-1]

    destination = "/".join([target,file.filename])
    file.save(destination)

    #TODO: remove hardcoding, currently processing image and storing it at /images as processed.jpg
    processImage(file)
    retObj = 0
    try:
       retObj =  predict(request.form['classifier'])
    except:
        retObj = predict()



    return retObj



if __name__ == "__main__":
    app.run()



