#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A = pd.read_csv("C:/Users/akaks/Downloads/Credit.csv")


# In[2]:


A.head()


# In[3]:


A.info()


# # Dropping Unwanted columns

# In[4]:


X = A.drop(labels=['Unnamed: 0','ID','Student'],axis=1)


# In[10]:


Y = A[['Student']]


# In[11]:


X


# # Preparing Y column

# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Ynew = le.fit_transform(Y)


# In[14]:


#Ynew


# # Preprocessing

# In[16]:


cat = []
con = []
for i in X.columns:
    if X[i].dtype=="O":
        cat.append(i)
    else:
        con.append(i)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)


# In[18]:


X2 = pd.get_dummies(X[cat])


# In[20]:


Xnew = X1.join(X2)


# In[21]:


Xnew


# In[22]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Ynew,test_size=0.2,random_state=21)


# In[23]:


Xnew.shape


# # Create a NN

# In[24]:


from keras.models import Sequential
from keras.layers import Dense


# In[39]:


#In output layer we will use activatin function as sigmoid


# In[25]:


nn = Sequential()
nn.add(Dense(14,input_dim=14))
nn.add(Dense(10))
nn.add(Dense(1,activation='sigmoid'))


# # Training the NN

# In[28]:


nn.compile(loss = 'binary_crossentropy',metrics='accuracy')
nn.fit(xtrain,ytrain,epochs=100)


# # Training accuracy

# In[30]:


Q = []
for i in nn.predict(xtrain):
    if(i<0.5):
        Q.append(0)
    else:
        Q.append(1)


# In[33]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(ytrain,Q)


# In[34]:


confusion_matrix(ytrain,Q)


# # Testing Accuracy

# In[35]:


Q = []
for i in nn.predict(xtest):
    if(i<0.5):
        Q.append(0)
    else:
        Q.append(1)


# In[36]:


confusion_matrix(ytest,Q)


# In[37]:


accuracy_score(ytest,Q)


# In[ ]:




