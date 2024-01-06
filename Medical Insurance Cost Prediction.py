#!/usr/bin/env python
# coding: utf-8

# In[179]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[180]:


get_ipython().system('pip install scikit-learn')


# In[181]:


#Importing required Libraries and Functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[182]:


#Loading the Dataset
insurance_dataset = pd.read_csv('Insurance.csv')


# In[183]:


insurance_dataset.head(10)


# In[184]:


#Rows and Columns
insurance_dataset.shape


# In[185]:


insurance_dataset.info()


# In[ ]:


#As we can see above, we have three categorical features in the dataset - Sex, Smoker, and Region


# In[186]:


#Checking for Missing Values
insurance_dataset.isnull().sum()


# In[187]:


#Data Analysis
insurance_dataset.describe()


# In[188]:


#Age Distribution
sns.set()
plt.figure(figsize=(6,6))
sns.histplot(insurance_dataset['age'],bins = 5)
plt.title('Age Distribution')
plt.show()


# In[189]:


#Gender Bifurcation
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()
insurance_dataset['sex'].value_counts()


# In[190]:


#BMI Distribution
#Normal BMI Range --> 18.5 to 24.9
plt.figure(figsize=(6,6))
sns.histplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[191]:


#Count of Children
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Number of Children')
plt.show()
insurance_dataset['children'].value_counts()


# In[192]:


#Count of Smokers
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('Smoker')
plt.show()
insurance_dataset['smoker'].value_counts()


# In[193]:


#Region Distribution
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('Region')
plt.show()
insurance_dataset['region'].value_counts()


# In[194]:


#Charges Distribution
plt.figure(figsize=(6,6))
sns.histplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# In[199]:


#Encoding the Categorical Variables
#Encoding 'sex' Column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

#Encoding 'smoker' Column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

#Encoding 'region' Column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# In[200]:


#Splitting Features and Targets
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[201]:


print(X)


# In[202]:


print(Y)


# In[203]:


#Splitting the Data into Training Data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[204]:


print(X.shape, X_train.shape, X_test.shape)


# In[205]:


#Model Training - Linear Regression
regressor = LinearRegression()


# In[206]:


regressor.fit(X_train, Y_train)


# In[207]:


#Model Evaluation
#Prediction on Training Data
training_data_prediction =regressor.predict(X_train)


# In[208]:


#R Squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[209]:


#Prediction on Test Data
test_data_prediction =regressor.predict(X_test)


# In[87]:


#R Squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# In[210]:


input_data = (31,1,25.74,0,1,0)

#Changing Data to a Numpy Array
input_data_as_numpy_array = np.asarray(input_data)

#Reshaping the Array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])


# In[ ]:





# In[ ]:





# In[ ]:




