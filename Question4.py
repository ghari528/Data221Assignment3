import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #had to use this
                                              # because I kept getting an ValueError: Input contains NaN
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

#y = loadKidneyDiseaseDataFrame["classification"]
#targetMatrix = loadKidneyDiseaseDataFrame.loc[:, ["classification"]]
targetMatrix = loadKidneyDiseaseDataFrame["classification"].str.strip().str.lower()
# *NEW* ensures 1D series removes DataConversionWarning. .str.strip() removes 'ckd\t'


#*NEW*
# Fill missing values

# Numeric columns: fill with median
numericColumns = featureMatrix.select_dtypes(include=['float64', 'int64']).columns
for col in numericColumns:
    featureMatrix[col] = featureMatrix[col].fillna(featureMatrix[col].median())

# Categorical columns: fill with mode
categoricalColumns = featureMatrix.select_dtypes(include=['object', 'string']).columns
for col in categoricalColumns:
    featureMatrix[col] = featureMatrix[col].fillna(featureMatrix[col].mode()[0])

# Encode categorical columns
for col in categoricalColumns:
    le = LabelEncoder()
    featureMatrix[col] = le.fit_transform(featureMatrix[col])


#Split Training Data(70%)
#Split Testing Data(30%)
#Use train_test_split with a fixed random_state
#random_state(test_size=0.3) leaving rest 70% for training
featureMatrix_train, featureMatrix_test, targetMatrix_train, targetMatrix_test = train_test_split(
    featureMatrix, targetMatrix, test_size=0.3)


# Train K-Nearest Neighbors classifier.
# Set number of neighbors to k = 5.
# Train the model using training data.
# Then use trained model to predict the labels of the test data.

#*NEW*
#before computing
# Import KNN
from sklearn.neighbors import KNeighborsClassifier

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
                                                    # It outputs predicted labels (ckd or notckd)
                                                    # It finds the 5 closest training points
                                                    # checks their labels
                                                    # predicts the majority label
                                                    # “Based on what I learned, here is my answer.”


##*NEW*
targetEncoder = LabelEncoder()
targetMatrix_train_encoded = targetEncoder.fit_transform(targetMatrix_train)
targetMatrix_test_encoded = targetEncoder.transform(targetMatrix_test)
predictions_encoded = targetEncoder.transform(predictions)


#After predictions
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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


# What True Positive, True Negative, False Positive, and False Negative
# mean in the context of kidney disease prediction?

# True Positive (TP) is when the model correctly predicts a patient
# has CKD.
# True Negative (TN) is when the model correctly predicts a patient
# does not have CKD.
# False Positive (FP) is when the model incorrectly predicts CKD for a
# patient who is healthy.
# False Negative (FN) is when the model incorrectly predicts no CKD
# for a patient who actually has CKD.


# Why accuracy alone may not be enough to evaluate a classification model?
# Accuracy is calculated as TP + TN / TP + TN + FP + FN.
# The matrix looks like [[TP FP],[FN TN]]

# Accuracy alone is not enough because it does not account for class imbalance.
# For example if most patients are healthy. It could fail to detect real CDK cases.
# Because a model predicting everyone as healthy may have high accuracy.
# But again, fail to catch the real CKD cases.


# Which metric is most important if missing a kidney disease case is
# very serious, and why?
# Recall investigates the power of the model in the detection of True
# responses: calculated as TP / TP + FN

# In this case missing a kidney disease case is very serious (False Negative).
# Recall is the most important metric. This is because it measures proportion
# of the actual CKD cases that this model correctly identifies. So we need
# to expect a high recall metric so patients with CKD are accounted for.
# This will reduce the risk of undiagnosed cases.
