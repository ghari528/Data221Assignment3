import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Load data
iris = datasets.load_iris()
loadKidneyDiseaseDataFrame = pd.DataFrame(data=iris.data, columns=iris.feature_names)
loadKidneyDiseaseDataFrame["classification"] = iris.target


#Create a matrix X that contains all columns except CKD.
#Note:To learn the model we have a data matrix X of variables
#     (features) values. For example number of rooms.
#     'y' would be house price, target variable.
#     These can be splitting to the training and testing sets using train_test_split function.

X = loadKidneyDiseaseDataFrame.drop(columns=["classification"])
featureMatrix = loadKidneyDiseaseDataFrame.loc[:, X.columns]


#Create a label vector y using CKD column

#y = loadKidneyDiseaseDataFrame["classification"]
#targetMatrix = loadKidneyDiseaseDataFrame.loc[:, ["classification"]]
targetMatrix = loadKidneyDiseaseDataFrame["classification"]


#*NEW*
# No missing values in Iris dataset, so no need to fill NaNs
# No categorical columns to encode in Iris dataset


#Split Training Data(70%)
#Split Testing Data(30%)
#Use train_test_split with a fixed random_state
#random_state(test_size=0.3) leaving rest 70% for training
featureMatrix_train, featureMatrix_test, targetMatrix_train, targetMatrix_test = train_test_split(
    featureMatrix, targetMatrix, test_size=0.3, random_state=42)

print("Feature Matrix Train:", featureMatrix_train.shape)
print("Feature Matrix Test:", featureMatrix_test.shape)
print("Target Matrix Train:", targetMatrix_train.shape)
print("Target Matrix Test:", targetMatrix_test.shape)
print(targetMatrix_train.value_counts())


# Train K-Nearest Neighbors classifier.
# Set number of neighbors to k = 5.
# Train the model using training data.
# Then use trained model to predict the labels of the test data.

#*NEW*
#before computing
# Create KNN model with k=5
knn_model = KNeighborsClassifier(n_neighbors=5)


# Train the model using training data only
knn_model.fit(featureMatrix_train, targetMatrix_train) #.fit() is where the model learns from the training data.
                                                       # Model looks at
                                                       # The input features (X) → featureMatrix_train
                                                       # The correct answers (y) → targetMatrix_train
                                                       # It learns the relationship between them(simply stores training data)
                                                       # Later uses distance to find 5 nearest neighbors
                                                       # "Study this data and remember the patterns"

# Predict using the test data
predictions = knn_model.predict(featureMatrix_test) # .predict() is where the model makes predictions on new data.
                                                    # The model takes the unseen test data
                                                    # It applies what it learned during .fit()
                                                    # It outputs predicted labels
                                                    # It finds the 5 closest training points
                                                    # checks their labels
                                                    # predicts the majority label
                                                    # “Based on what I learned, here is my answer.”




#After predictions

# Compute and display the confusion matrix
confusionMatrixResult = confusion_matrix(targetMatrix_test, predictions)
print(f"Confusion Matrix:\n{confusionMatrixResult}")

# Compute and print Accuracy
modelAccuracy = accuracy_score(targetMatrix_test, predictions)
print(f"Model Accuracy: {modelAccuracy}")

# Compute and print Precision
modelPrecision = precision_score(targetMatrix_test, predictions, average='weighted')
print(f"Model Precision: {modelPrecision:}")

# Compute and print Recall
modelRecall = recall_score(targetMatrix_test, predictions, average='weighted')
print(f"Model Recall: {modelRecall}")

# Compute and print F1-score
modelF1Score = f1_score(targetMatrix_test, predictions, average='weighted')
print(f"Model F1 Score: {modelF1Score}")


#How changing k affects the behavior of the model
# Changing the value of k changes how many nearest neighbors the
# model uses when making predictions.

#Why very small values of k may cause overfitting
# Small k makes model very sensitive to individual data points.
# Small k like 1 or 3 only looks at 1 or 3 closest neighbors.
# That means predictions relies heavily on just a few points.
# If one of those neighbors is an outlier it can change the
# predictions by a lot. This can lead to overfitting.

#Why very large values of k may cause underfitting
# Larger k values result in smoother and more generalized decision boundaries.
# In this way the outliers have less influence but can miss the small patterns
# if k is too large. This leads to underfitting.
