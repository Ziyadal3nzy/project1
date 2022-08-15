#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[32]:


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Our Goal: to predict the price given a set of attributes.

# ## Load the Data

# In[2]:


from sklearn import datasets
boston = datasets.load_boston()


# In[3]:


X_boston,y_boston = boston.data, boston.target
print('Shape of data:', X_boston.shape, y_boston.shape)


# In[4]:


print('Keys:', boston.keys())
print('Feature names:',boston.feature_names)


# In[5]:


boston.target


# In[6]:


print(boston.DESCR)


# ## EDA (Exploratory Data Analysis)

# ## Q1: Create a dataframe and Save that dataset inside it.

# In[7]:


USAhousing = pd.read_csv('USA_Housing.csv')


# In[8]:


df = pd.DataFrame(boston.data, columns = boston.feature_names)
df


# ## Q2: Print the head rows of the dataframe.

# In[9]:


df['Price'] = boston.target
df


# ## Q3: Use histogram to show the distribution of House Prices.

# In[10]:


sns.distplot(df['Price'])


# ## Q4: Use a heatmap to show the correlation between features and the target labels.

# In[11]:


sns.heatmap(df.corr(), annot=True);


# ## Q5: Use a lmplot to draw the relations between price and LSTAT.

# In[12]:


sns.lmplot(data=df, x='Price', y='LSTAT')


# ## Q6: Use a lmplot to draw the relations between price and RM.

# In[13]:


sns.lmplot(data=df, x='Price', y='RM')


# In[16]:


df.columns


# ## Q7: Split the dataset into Train and Test sets with test_size=30% and random_state=23.

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X = df[['CRIM', 'ZN', 'INDUS','CHAS', 'NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
y = df['Price']


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)


# ## Q8: Build a Linear Regression Model.

# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


lm = LinearRegression()


# ## Q9: Train the Model.

# In[24]:


lm.fit(X_train,y_train)


# ## Q10: Evaluate the model. 
# - print intercept and coefficients.
# - compare between predictions and real values, then visualize them.
# - Draw Residual Histogram.

# In[25]:


print(lm.intercept_)


# In[26]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[27]:


predictions = lm.predict(X_test)
predictions


# In[28]:


Real_Values = np.array(y_test)
Real_Values


# In[ ]:





# ## Residual Histogram

# In[29]:


sns.distplot((lm.intercept_-coeff_df),bins=50);


# ## Q11: Use evaluation metrics MAE, MSE, RMSE and R^2.

# In[30]:


from sklearn import metrics


# In[31]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[33]:


print('R^2', metrics.r2_score(Real_Values, predictions))

