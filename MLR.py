#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression Model (Combined Cycle Power Plant Data Set)

# # Import Liabraries
# 

# In[10]:


import pandas as pd
import numpy as np


# # Import dataset

# In[19]:


data_df=pd.read_excel('CCPP\dataset.xlsx')


# In[20]:


data_df.head()


# # Define X and Y

# In[21]:


x=data_df.drop(['PE'], axis=1).values
y=data_df['PE'].values


# In[22]:


print(x)


# In[23]:


print(y)


# # Split the dataset in training set and test set

# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# # Train the model on the training set

# In[28]:


from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)


# # Predict the test set results

# In[29]:


y_pred=ml.predict(x_test)
print(y_pred)


# In[31]:


ml.predict([[14.96,41.76,1024.07,73.17]])


# # Evaluate the model
# 

# In[32]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# # Plot the result
# 

# In[35]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# # Predicted values

# In[37]:


pred_y_df=pd.DataFrame({'Actual Value':y_test, 'Predicted value':y_pred, 'Difference':y_test-y_pred})
pred_y_df[0:30]

