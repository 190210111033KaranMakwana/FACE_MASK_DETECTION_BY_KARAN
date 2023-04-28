#!/usr/bin/env python
# coding: utf-8

# In[157]:


get_ipython().system('pip install opencv-python')


# In[158]:


import cv2


# In[159]:


image = cv2.imread('mask14.jpg')


# In[160]:


image.shape


# In[161]:


image[0]


# In[162]:


import matplotlib.pyplot as plt


# In[163]:


plt.imshow(image)


# In[164]:


while True:
    cv2.imshow('result',image)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[165]:


haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[166]:


haar_data.detectMultiScale(image)


# In[ ]:





# In[167]:


# cv2.rectangular(img,(x,y),(w,h),(b,g,r),border_thickness)


# In[168]:


while True:
    faces = haar_data.detectMultiScale(image)
    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (y+w, x+h), (255,0,255), 4)
    cv2.imshow('result',image)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[ ]:





# In[277]:


capture = cv2.VideoCapture(0)   # to initiaize camera
data = []                       # to store face data
while True:
    flag, image = capture.read()    # read video frame by frame & return T/F and one frame at a time  
    
    if flag:                    # will check if flag is true
        faces = haar_data.detectMultiScale(image)   # detecting face from the frame
        for x,y,w,h in faces:                       # fetching x, y, w, h of face dectected in frame  
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 4)   # drawing rectange on face
            face = image[y: y+h, x:x+w, :]         # slicing only face from the frame 
            face = cv2.resize(face, (50, 50))    #resizeing all faces to 50 x 50
            print(len(data))
            if len(data) < 400:        # condition for only storing 400 images
                data.append (face)    # storing face data
        cv2.imshow('result',image)     # to show the window
    
        if cv2.waitKey(2) == 27 or len(data) >= 200:   # break loop if escaped is pressed or 200 faces are stored
            break
        
capture.release()         # release the camera object holded by openCV
cv2.destroyAllWindows()    # close all the window opened by openCV


# In[ ]:





# In[ ]:





# In[246]:


import numpy as np


# In[26]:


x = np.array([3,2,54,6])


# In[27]:


x


# In[28]:


x[0:2]


# In[29]:


x = np.array([[3,4,54,67,8,8],[1,2,2,4,5,7],[4,5,3,5,6,7],[1,2,3,34,6,8]])


# In[30]:


x


# In[31]:


x[0]


# In[32]:


x[0][1:4]


# In[33]:


x[0:3, 0:3]


# In[34]:


x[:, 1:4]


# In[ ]:





# In[243]:


np.save('with_mask.npy', data)


# In[244]:


plt.imshow(data[0])


# In[247]:


np.save('without_mask.npy', data)


# In[248]:


plt.imshow(data[0])


# In[ ]:


np.save('with_mask.npy', data)


# In[ ]:


plt.imshow(data[0])


# In[66]:


capture = cv2.VideoCapture(0)
data = []
while True:
    flag, image = capture.read()
    
    if flag:
        faces = haar_data.detectMultiScale(image)
    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 4)
        face = image[y: y+h, x:x+w, :]
        face = cv2.resize(face, (50, 50))
        print(len(data))
        if len(data) < 400:
            data.append (face)
    cv2.imshow('result',image)
    
    if cv2.waitKey(2) == 27 or len(data) >= 200:
        break
        
capture.release()
cv2.destroyAllWindows()


# In[ ]:


np.save('with_mask.npy', data)


# In[ ]:


plt.imshow(data[0])


# In[249]:


import numpy as np
import cv2


# In[250]:


with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')


# In[251]:


with_mask.shape


# In[252]:


without_mask.shape


# In[253]:


with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)


# In[254]:


with_mask.shape


# In[255]:


without_mask.shape


# In[256]:


X = np.r_[with_mask, without_mask]


# In[257]:


X.shape


# In[258]:


labels = np.zeros(X.shape[0])


# In[259]:


labels[200:] = 1.0


# In[260]:


names = {0: 'Mask', 1: 'No Mask'}


# In[ ]:





# In[261]:


#svm and Svc model


# In[262]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


# In[263]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[264]:


x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)


# In[265]:


x_train


# In[266]:


x_train.shape


# In[267]:


from sklearn.decomposition import PCA


# In[268]:


pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)


# In[269]:


x_train[0]


# In[270]:


x_train.shape


# In[271]:


x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)


# In[272]:


svm = SVC()
svm.fit(x_train, y_train)


# In[273]:


# x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[274]:


accuracy_score(y_test, y_pred)


# In[275]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[276]:


haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    flag, image = capture.read()
    
    if flag:
        faces = haar_data.detectMultiScale(image)
        for x,y,w,h in faces:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 4)
            face = image[y: y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(image, n, (x,y), font, 1, (244,250,250), 2)
            print(n)
        cv2.imshow('result',image)
    
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break
        
capture.release()
cv2.destroyAllWindows()


# In[ ]:




