# Prediction-Analysis-of-Housing-Price-in-Melbourne-using-Machine-and-Deep-Learning-on-Python

The Project involves predictive analysis of Housing prices in Melbourne based on all the given factors in the dataset.
The dataset is first of all cleaned and then EDA is performed. 

we split our data into training and testing data 80% for training and 20% for testing. After that we normalize our training dataset, 
to normalize we take mean of our training data and minus original values from mean values then divide by standard deviation of data. 
The same steps we done with testing data to get both training and testing datasets normalized.


According to business analysis the most important field was price. In a sense that we take all the other inputs and then predict the price for 
that reason we select our best fields those are affecting our dataset the most includes Suburb, Type, Method, SellerG, Distance, Postcode, 
Building area, Year building, Council area, Region area and Property count.   

After that we have apply regression techniques and use 3 techniques:
•	Linear Regression
•	Neural Network
•	XGBoost
Then we compare the accuracy of above 3 techniques and neural network and xgboost accuracy are high
