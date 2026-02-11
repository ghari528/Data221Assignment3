import pandas as pd
from sklearn.model_selection import train_test_split

#load data frame from csv first
loadKidneyDiseaseDataFrame = pd.read_csv('kidney_disease.csv')


#Create a matrix X that contains all columns except CKD.
#Note:To learn the model we have a data matrix X of variables
#     (features) values. For example number of rooms.
#     'y' would be house price, target variable.
#     These can be splitting to the training and testing sets using train_test_split function.

X = loadKidneyDiseaseDataFrame.drop(columns=["classification"])
featureMatrix = loadKidneyDiseaseDataFrame.loc[:, X.columns]



#Create a label vector y using CKD column

y = loadKidneyDiseaseDataFrame["classification"]
targetMatrix = loadKidneyDiseaseDataFrame.loc[:, ["classification"]]


#Split Training Data(70%)
#Split Testing Data(30%)
#Use train_test_split with a fixed random_state
#random_state(test_size=0.3) leaving rest 70% for training
featureMatrix_train, featureMatrix_test, targetMatrix_train, targetMatrix_test = train_test_split(
    featureMatrix, targetMatrix, test_size=0.3)


print("Feature Matrix Train:", featureMatrix_train.shape)
print("Feature Matrix Test:", featureMatrix_test.shape)
print("Target Matrix Train:", targetMatrix_train.shape)
print("Target Matrix Test:", targetMatrix_test.shape)
print(targetMatrix_train["classification"].value_counts())


#Note: Train/Test is a method to measure the accuracy of your model.
#      It is called Train/Test because you split the data set into
#      two sets: a training set and a testing set.You train the model
#      using the training set. You test the model using the testing set.
#      Train the model means create the model.
#      Test the model means test the accuracy of the model.



# Why we should not train and test a model on the same data(overfitting and underfitting)

# If we train and test on the same data then the model will
# memorize the training points rather than learning patterns.

# This can lead to overfitting where the model does well on training
# data but fails to be representative of the new data added to the plot.

# This means the model may look good on the training set but it is
# not generalizable for new data (can't make accurate predictions on real world data).
# Even a simple model that seems to fit the training points well still
# may not represent new data correctly if we only test on what it has already seen.
#So training and testing on the same data doesn't give realistic measure of model performance.


# What the purpose of the testing set?

# The training set is a portion of the data set that is other than
# the training set. So part of the data is held out as the testing set
# to check performance of data. What this does is check how well
# the model generalizes to new, unseen data. By checking the performance
# on the testing set we can see if the model is overfitted
# (too complex, fits only training data) or if it is
# underfitted (too simple, cannot capture patterns in the data).
# This solution gives an objective check to make sure model works on new data.