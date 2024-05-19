#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import math


# In[2]:


data = loadmat('mnist_all.mat')


# In[3]:


train0 = data['train0']
train1 = data['train1']
train2 = data['train2']
train3 = data['train3']
train4 = data['train4']
train5 = data['train5']
train6 = data['train6']
train7 = data['train7']
train8 = data['train8']
train9 = data['train9']


# In[4]:


test0 = data['test0']
test1 = data['test1']
test2 = data['test2']
test3 = data['test3']
test4 = data['test4']
test5 = data['test5']
test6 = data['test6']
test7 = data['test7']
test8 = data['test8']
test9 = data['test9']


# In[5]:


data_0 = pd.DataFrame(train0)
label=[0]*data_0.shape[0]
data_0.insert(0,'label',label)
data_1 = pd.DataFrame(train1)
label=[1]*data_1.shape[0]
data_1.insert(0,'label',label)

data_2 = pd.DataFrame(train2)
label=[2]*data_2.shape[0]
data_2.insert(0,'label',label)
data_3 = pd.DataFrame(train3)
label=[3]*data_3.shape[0]
data_3.insert(0,'label',label)

data_4 = pd.DataFrame(train4)
label=[4]*data_4.shape[0]
data_4.insert(0,'label',label)
data_5 = pd.DataFrame(train5)
label=[5]*data_5.shape[0]
data_5.insert(0,'label',label)

data_6 = pd.DataFrame(train6)
label=[6]*data_6.shape[0]
data_6.insert(0,'label',label)
data_7 = pd.DataFrame(train7)
label=[7]*data_7.shape[0]
data_7.insert(0,'label',label)

data_8 = pd.DataFrame(train8)
label=[8]*data_8.shape[0]
data_8.insert(0,'label',label)
data_9 = pd.DataFrame(train9)
label=[9]*data_9.shape[0]
data_9.insert(0,'label',label)


# In[6]:


data_test_0 = pd.DataFrame(test0)
label=[0]*data_test_0.shape[0]
data_test_0.insert(0,'label',label)
data_test_1 = pd.DataFrame(test1)
label=[1]*data_test_1.shape[0]
data_test_1.insert(0,'label',label)

data_test_2 = pd.DataFrame(test2)
label=[2]*data_test_2.shape[0]
data_test_2.insert(0,'label',label)
data_test_3 = pd.DataFrame(test3)
label=[3]*data_test_3.shape[0]
data_test_3.insert(0,'label',label)

data_test_4 = pd.DataFrame(test4)
label=[4]*data_test_4.shape[0]
data_test_4.insert(0,'label',label)
data_test_5 = pd.DataFrame(test5)
label=[5]*data_test_5.shape[0]
data_test_5.insert(0,'label',label)

data_test_6 = pd.DataFrame(test6)
label=[6]*data_test_6.shape[0]
data_test_6.insert(0,'label',label)
data_test_7 = pd.DataFrame(test7)
label=[7]*data_test_7.shape[0]
data_test_7.insert(0,'label',label)

data_test_8 = pd.DataFrame(test8)
label=[8]*data_test_8.shape[0]
data_test_8.insert(0,'label',label)
data_test_9 = pd.DataFrame(test9)
label=[9]*data_test_9.shape[0]
data_test_9.insert(0,'label',label)


# In[7]:


data_train = [data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9]
data_train = pd.concat(data_train)


# In[8]:


data_test = [data_test_0,data_test_1,data_test_2,data_test_3,data_test_4,data_test_5,data_test_6,data_test_7,data_test_8,data_test_9]
data_test = pd.concat(data_test)


# In[9]:


data_train = np.array(data_train)
m,n = data_train.shape
np.random.shuffle(data_train)

data_train  = data_train.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

data_test = np.array(data_test)
data_test = data_test.T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255


# In[10]:


def init_params(size):
    W1 = np.random.normal(size=(100,size)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(100, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 100)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    
    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1,Y.size))
    one_hot_Y[Y,np.arange(Y.size)] = 1
    return one_hot_Y


def derive_ReLU(Z):
    return Z > 0

def back_prop(Z1 ,A1,A2,W2,X,Y,m):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/ m *dZ2.dot(A1.T)
    db2 = 1/m *np.sum(dZ2,axis=1,keepdims = True)
    dZ1 = W2.T.dot(dZ2) * derive_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/ m *np.sum(dZ1,axis=1,keepdims = True)
    return dW1,db1,dW2,db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha *dW1
    b1 = b1 - alpha *db1
    W2 = W2 - alpha *dW2
    b2 = b2 -alpha *db2
    return W1,b1,W2,b2


# In[11]:


def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X ,Y ,iterations):
    size , m =X.shape
    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward_prop(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2 = back_prop(Z1,A1,A2,W2,X,Y,m)
        alpha = 1/ math.sqrt(i+1)
        W1,b1,W2,b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if((i+1) % int(iterations / 10) == 0):
            print("iteration: " ,i+1)
            predictions = get_predictions(A2)
            print("Accuracy: ",get_accuracy(predictions,Y))
    return W1,b1,W2,b2


# In[12]:


W1,b1,W2,b2 = gradient_descent(X_train,Y_train,1000)


# In[13]:


def make_predictions(X,W1,b1,W2,b2):
    _,_,_,A2 = forward_prop(W1,b1,W2,b2,X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index,W1,b1,W2,b2):
    vect_X = X_train[:,index,None]
    print(vect_X.shape)
    prediction = make_predictions(vect_X,W1,b1,W2,b2)
    label = Y_train[index]
    
    print("prediction: ",prediction)
    print("label: ",label)
    
    current_image = vect_X.reshape((28,28))*255
    plt.pyplot.gray()
    plt.pyplot.imshow(current_image,interpolation = 'nearest')
    plt.pyplot.show()


# In[ ]:





# In[14]:


test_prediction(108,W1,b1,W2,b2)


# In[15]:


dev_predictions = make_predictions(X_test, W1, b1, W2, b2)
get_accuracy(dev_predictions,Y_test)


# In[16]:


import pickle 
with open("trained_params.pkl", "wb") as dump_file:
    pickle.dump((W1, b1, W2, b2), dump_file)


# In[ ]:




