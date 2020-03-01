# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:21:22 2020

@author: prasa
"""

import numpy as np
import pandas as pd
import os
import matplotlib
from tqdm import tqdm_notebook
import cv2 as cv
import glob
import matplotlib as plt

class ObjectDetector():
    '''
    Class for Object Detection
    '''
    def __init__(self,object_cascade_path):
        '''
        param: object_cascade_path - path for the *.xml defining the parameters for {face, eye, smile, profile}
        detection algorithm
        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades
        '''

        self.objectCascade=cv.CascadeClassifier(object_cascade_path)


    def detect(self, image, scale_factor=1.3,
               min_neighbors=5,
               min_size=(20,20)):
        '''
        Function return rectangle coordinates of object for given image
        param: image - image to process
        param: scale_factor - scale factor used for object detection
        param: min_neighbors - minimum number of parameters considered during object detection
        param: min_size - minimum size of bounding box for object detected
        '''
        rects=self.objectCascade.detectMultiScale(image,
                                                scaleFactor=scale_factor,
                                                minNeighbors=min_neighbors,
                                                minSize=min_size)
        return rects
    

def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
    
def detect_face(net, img):
    
    h, w = img.shape[:2]
    
    img = img - 127.5
    img = img / 127.5
    img = img.astype(np.float32)
    
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), swapRB=False)
    net.setInput(blob)
    objectsModelPreds = net.forward()
#    print("objectsModelPreds>>>>>>>", objectsModelPreds)

    clsn = objectsModelPreds[0,0,:,1]
    conf = objectsModelPreds[0,0,:,2]
    box = objectsModelPreds[0,0,:,3:7]* np.array([w, h, w, h])
    box, conf, cls_name = (box.astype(np.int32), conf, clsn)
        
    return box, conf, cls_name, objectsModelPreds

def get_meta_from_json(json_path):
    df = pd.read_json(json_path)
    df = df.T
    return df

def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))

def display_image_from_video(video_path,detector_net):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    while ret == True:
        ret, frame = capture_image.read()
        face_bbox, conf, label, objectsModelPreds = detect_face(detector_net, frame)

        
        cv.imshow('frame ', face_bbox)
        key = cv.waitKey(0) & 0xFF
         
        if key == ord("q"):
            cv.destroyAllWindows()


def add_padding(bboxes, pad_val, bg_h, bg_w):
    
    new_boxes = []
    for box in bboxes:
        
        x_min, y_min, x_max, y_max = box
        
        x_min -= int(x_min * pad_val) if x_min * pad_val > 0 else 1
        y_min -= int(y_min * pad_val) if y_min * pad_val > 0 else 1
        
        x_max += int(x_max * pad_val) if x_max * pad_val <= bg_w else bg_w
        
        y_max += int(y_max * (pad_val + 0.1)) if y_max * (pad_val + 0.1) <= bg_h else bg_h
        
        new_boxes.append([x_min, y_min, x_max, y_max])
    
    return new_boxes

def detect_objects(image, scale_factor, min_neighbors, min_size):
    '''
    Objects detection function
    Identify frontal face, eyes, smile and profile face and display the detected objects over the image
    param: image - the image extracted from the video
    param: scale_factor - scale factor parameter for `detect` function of ObjectDetector object
    param: min_neighbors - min neighbors parameter for `detect` function of ObjectDetector object
    param: min_size - minimum size parameter for f`detect` function of ObjectDetector object
    '''
    
    image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    eyes=ed.detect(image_gray,
                   scale_factor=scale_factor,
                   min_neighbors=min_neighbors,
                   min_size=(int(min_size[0]/2), int(min_size[1]/2)))

    for x, y, w, h in eyes:
        #detected eyes shown in color image
        cv.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 0,255),3)
 
    # deactivated due to many false positive
    #smiles=sd.detect(image_gray,
    #               scale_factor=scale_factor,
    #               min_neighbors=min_neighbors,
    #               min_size=(int(min_size[0]/2), int(min_size[1]/2)))

    #for x, y, w, h in smiles:
    #    #detected smiles shown in color image
    #    cv.rectangle(image,(x,y),(x+w, y+h),(0, 0,255),3)


    profiles=face_pd.detect(image_gray,
                   scale_factor=scale_factor,
                   min_neighbors=min_neighbors,
                   min_size=min_size)

    for x, y, w, h in profiles:
        #detected profiles shown in color image
        cv.rectangle(image,(x,y),(x+w, y+h),(255, 0,0),3)

    faces=fd.detect(image_gray,
                   scale_factor=scale_factor,
                   min_neighbors=min_neighbors,
                   min_size=min_size)

    for x, y, w, h in faces:
        #detected faces shown in color image
        cv.rectangle(image,(x,y),(x+w, y+h),(0, 255,0),3)
    

    # image
    cv.imshow('corped image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def extract_image_objects(video_file):
    '''
    Extract one image from the video and then perform face/eyes/smile/profile detection on the image
    param: video_file - the video from which to extract the image from which we extract the face
    '''
    video_path = 'train_sample_videos/' + video_file
    capture_image = cv.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    detect_objects(image=frame, 
            scale_factor=1.3, 
            min_neighbors=5, 
            min_size=(50, 50))
    
TRAIN_SAMPLE_FOLDER = glob.glob('train_sample_videos/*')
TEST_FOLDER = glob.glob('test_videos/*')

train_list = TRAIN_SAMPLE_FOLDER
json_path = 'train_sample_videos/metadata.json'

detector_model = "face_detector/best_bn_full.caffemodel"
detector_config = "face_detector/ssd_face_deploy_bn.prototxt"
detector_net = cv.dnn.readNet(detector_model, detector_config)

meta_train_df = get_meta_from_json(json_path)
meta_train_df.head()

print(missing_data(meta_train_df))
print(most_frequent_values(meta_train_df))

meta = np.array(list(meta_train_df.index))
storage = np.array([file for file in train_list if  file.endswith('mp4')])
print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")
print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")
print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")


#finding the fake videos for 3 sample

fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].index)
fake_train_sample_video
real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].index)
real_train_sample_video


#for video_file in fake_train_sample_video:
##    display_image_from_video('train_sample_videos/' + video_file ,detector_net)
#    capture_image = cv.VideoCapture('train_sample_videos/' + video_file) 
#    ret, frame = capture_image.read()
#    
#    while True:
#        ret, frame = capture_image.read()
#        if ret == False:
#            break
#        
#        bg_h, bg_w = frame.shape[:-1]
#        face_bbox, conf, label, objectsModelPreds = detect_face(detector_net, frame)
#        face_bbox = add_padding(face_bbox, 0.005, bg_h, bg_w)
#
#        for i in range(len(face_bbox)):
#            
#            if conf[i] <= 0.1:
#                continue
#            
#            x_min, y_min, x_max, y_max = face_bbox[i]
#    
#            label_conf = "face_{:.2f}".format(conf[i])
#            face_image = frame[y_min : y_max, x_min : x_max]
#            
#        cv.imshow('frame ', face_image)
#        key = cv.waitKey(1) & 0xFF
#         
#        if key == ord("q"): 
#            cv.destroyAllWindows()


#Frontal face, profile, eye and smile  haar cascade loaded
            
frontal_cascade_path= 'haar-cascades-for-face-detection/haarcascade_frontalface_default.xml'
eye_cascade_path= 'haar-cascades-for-face-detection/haarcascade_eye.xml'
profile_cascade_path= 'haar-cascades-for-face-detection/haarcascade_profileface.xml'
smile_cascade_path='haar-cascades-for-face-detection/haarcascade_smile.xml'

#Detector object created
# frontal face
fd=ObjectDetector(frontal_cascade_path)
# eye
ed=ObjectDetector(eye_cascade_path)
# profile face
face_pd =ObjectDetector(profile_cascade_path)
# smile
sd=ObjectDetector(smile_cascade_path)

same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='kgbkktcjxf.mp4'].index)
for video_file in same_original_fake_train_sample_video[1:4]:
    print(video_file)
    extract_image_objects(video_file)






