#!/usr/bin/env python
# coding: utf-8

# # REGRESSION ON PRICES

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt


# Load data
dataset = pd.read_csv('melbourne_house_data.csv')



# In[4]:


#STEP 1
#UNDERSTAND DATA AND VISUALIZING
dataset.head()


# In[6]:


#visualizing suburb
dataset['Suburb'].value_counts().head(50).plot.bar()


# In[7]:


#bar plot for seller g
dataset['SellerG'].value_counts().head(50).plot.bar()


# In[179]:


#line plot for prices
dataset['Price'].value_counts().sort_index().plot.line()


# In[181]:


#describing data viewing mean , count , deviation , min ,max
dataset.describe()


# In[182]:


#viewing null values
dataset.isnull().sum()

# Percentage of missing values
dataset.isnull().sum()/len(dataset)*100


# In[184]:


#step 2
#filling noncategorical values with mean values
dataset.fillna(dataset.mean(),inplace=True)


# In[185]:


dataset.isnull().sum()
dataset.isnull().sum()/len(dataset)*100


# In[186]:


# Remove rows missing data
dataset = dataset.dropna()

# Confirm that observations missing data were removed  
dataset.info()


# In[187]:


# Removing values in which building area = 0 as it is a outlier
dataset = dataset[dataset['BuildingArea']!=0]


# In[188]:


#importing labelencoder to encode categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

#converting categorical values to numerica values
dataset.iloc[:, 1] = labelencoder.fit_transform(dataset.iloc[:, 1])
dataset.iloc[:, 4] = labelencoder.fit_transform(dataset.iloc[:, 4])
dataset.iloc[:, 6] = labelencoder.fit_transform(dataset.iloc[:, 6])
dataset.iloc[:, 7] = labelencoder.fit_transform(dataset.iloc[:, 7])
dataset.iloc[:, 17] = labelencoder.fit_transform(dataset.iloc[:, 17])
dataset.iloc[:, 20] = labelencoder.fit_transform(dataset.iloc[:, 20])

display(dataset)


# In[144]:


# Split
# selecting independent variable 
X =dataset[['Suburb','Rooms','Type' ,'Method','SellerG','Distance', 'Postcode','Bathroom', 'Car', 'Landsize', 
            'BuildingArea', 'YearBuilt','CouncilArea','Regionname','Propertycount']]

# selecting dependent variable
y = dataset['Price']

# Train, test, split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)


# In[153]:


#normalizing data
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std


# # performing regression using 3 techniques
# STEP 3

# In[154]:


#Linear regression
# Fit
# Import linear regression model
from sklearn.linear_model import LinearRegression

# Create linear regression object
regressor = LinearRegression()

# Fit model to training data
regressor.fit(X_train,y_train)

# Predict
# Predicting test set results
linear_regression_pred = regressor.predict(X_test)


# In[156]:


#neural network


from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
input_dim=15,))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train,y_train,batch_size=32,epochs=100)

# Predicting test set results
neural_network_pred= model.predict(X_test)


# In[158]:


from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

# Predicting test set results
XGboost_pred= clf.predict(X_test)


# In[161]:

plt.scatter(X_train[1],y_train,color = 'red')
plt.plot(X_test[1],regressor.predict(X_test),color = 'blue')
plt.title('Amount V X')
plt.xlabel('X')
plt.ylabel('Amount in $')
plt.show()

comparison={"actual":[],"linear_regression":[],"XGboost":[] }
for i in range (len(y_test)):
    ABC
    comparison['actual'].append(y_test.values[i])
    comparison['linear_regression'].append(int(linear_regression_pred[i]))
    #comparison["neural_network"].append(int(neural_network_pred[i][0]))
    comparison['XGboost'].append(int(XGboost_pred[i]))
final = pd.DataFrame(comparison,)
display(final)


plt.scatter(comparison['linear_regression'],color = 'blue')
plt.plot(comparison['actual'],color = 'green')
plt.plot(comparison['XGboost'],color = 'red')
plt.ylabel('Amount in $')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)



