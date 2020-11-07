#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[31]:


data=pd.read_csv(r'C:\Users\Administrator\Desktop\ML\UPGRAD\Housing.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[7]:


data.describe()


# In[4]:


sns.pairplot(data)


# In[5]:


plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x='airconditioning',y='price',data=data)
plt.subplot(2,3,2)
sns.boxplot(x='mainroad',y='price',data=data)
plt.subplot(2,3,3)
sns.boxplot(x='guestroom',y='price',data=data)
plt.subplot(2,3,4)
sns.boxplot(x='basement',y='price',data=data)
plt.subplot(2,3,5)
sns.boxplot(x='hotwaterheating',y='price',data=data)


# In[32]:


varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function

data[varlist]=data[varlist].apply(lambda x: x.map({"yes": 1, "no": 0}))    


# In[33]:


data.head()


# In[34]:


cat=pd.get_dummies(data["furnishingstatus"],drop_first=True)


# In[16]:


cat.head()


# In[35]:


data=pd.concat([data,cat],axis=1)


# In[8]:


data


# In[36]:


data.drop(['furnishingstatus'], axis = 1, inplace = True)


# In[37]:


data


# In[38]:


df_train, df_test = train_test_split(data, train_size = 0.7, test_size = 0.3, random_state = 100)



# In[39]:


df_train.head()


# In[40]:


#creating the train test split and rescaling
scaler = MinMaxScaler()
num_vars=['price','area','bedrooms','bathrooms','stories','parking']
df_train[num_vars]=scaler.fit_transform(df_train[num_vars])


# In[41]:


df_train.head()


# In[42]:


df_train


# In[43]:


plt.figure(figsize = (6, 6))
plt.scatter(df_train.area, df_train.price)
plt.show()


# In[44]:



y_train = df_train.pop('price')
y_train


# In[45]:


X_train = df_train
X_train


# In[46]:


y_train


# In[47]:


X_train_lm=sm.add_constant(X_train['area'])


# In[48]:


lr = sm.OLS(y_train, X_train_lm).fit()


# In[49]:


lr.params


# In[50]:


lr.summary()


# In[51]:


X_train_lm = X_train[['area', 'bathrooms','bedrooms']]

X_train_lm = sm.add_constant(X_train_lm)

lr = sm.OLS(y_train, X_train_lm).fit()

lr.params


# In[52]:


lr.summary()


# In[53]:


X_train.columns


# In[54]:


X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
lr.params
lr.summary()


# In[55]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[56]:


X=X_train.drop('semi-furnished',1)


# In[57]:


X


# In[60]:


X_train_lm=sm.add_constant(X)
lr=sm.OLS(y_train,X_train_lm).fit()
lr.summary()


# In[61]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[77]:


X=X.drop('bedrooms',1)


# In[78]:


X_train_lm=sm.add_constant(X)
lr=sm.OLS(y_train,X_train_lm).fit()
lr.summary()


# In[82]:


vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[83]:


X=X.drop('basement',1)


# In[84]:





# In[85]:


X_train_lm=sm.add_constant(X)
lr=sm.OLS(y_train,X_train_lm).fit()
lr.summary()


# In[86]:


vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[88]:


y_train_pred=lr.predict(X_train_lm)
y_train_pred


# In[90]:


fig=plt.figure()
sns.distplot(y_train - y_train_pred, bins =20)
fig.suptitle('Error Terms', fontsize = 20)                
plt.xlabel('Errors', fontsize = 18)        


# In[94]:


df_test.head()


# In[95]:


num_vars=['price','area','bedrooms','bathrooms','stories','parking']
df_test[num_vars]=scaler.transform(df_test[num_vars])


# In[97]:


df_test.describe()


# In[101]:


y_test=df_test.pop('price')
y_test


# In[116]:


X_test = df_test

X_test_lm=sm.add_constant(X_test)
X_test_lm=X_test_lm.drop(['bedrooms','basement','semi-furnished'],1)

X_test_lm


# In[121]:


y_test_pred = lr.predict(X_test_lm)
y_test_pred


# In[130]:


fig=plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle("y_test vs y_pred")
plt.xlabel('y_test', fontsize = 18)                          
plt.ylabel('y_pred', fontsize = 16) 


# In[124]:


lr.summary()


# In[ ]:




