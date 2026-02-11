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
X = loadKidneyDiseaseDataFrame.drop(columns=["classification"])
featureMatrix = loadKidneyDiseaseDataFrame.loc[:, X.columns]


#Create a label vector y using CKD column
targetMatrix = loadKidneyDiseaseDataFrame["classification"]


#Train/Test Split (70/30)
featureMatrix_train, featureMatrix_test, targetMatrix_train, targetMatrix_test = train_test_split(
    featureMatrix, targetMatrix, test_size=0.3, random_state=42)

#*NEW*
#Providied k values
kValues = [1, 3, 5, 7, 9]
kAccuracyResults = []
#will look like [[1,current accuracy],[3,..],[]..]

#Use for loop because need to loop through different values of k
for currentKValue in kValues:
    temporaryKnnModel = KNeighborsClassifier(n_neighbors=currentKValue)

    #Train model
    temporaryKnnModel.fit(featureMatrix_train, targetMatrix_train)

    #Predict
    predictedLabels = temporaryKnnModel.predict(featureMatrix_test)

    #Calculates accuracy if k value using accuracy_score()
    currentAccuracy = accuracy_score(targetMatrix_test, predictedLabels)

    kAccuracyResults.append((currentKValue, currentAccuracy))

    print(f"k = {currentKValue} \n Accuracy = {currentAccuracy:.4f}")

#Find best k

bestK = kAccuracyResults[0][0]
highestAccuracy = kAccuracyResults[0][1]

for kValue, accuracyValue in kAccuracyResults:
    if accuracyValue > highestAccuracy:
        bestK = kValue
        highestAccuracy = accuracyValue
print(f"Best k value. k = {bestK} Accuracy = {highestAccuracy:}.")


#Evaluate model using best k
bestKnn_model = KNeighborsClassifier(n_neighbors=bestK)
bestKnn_model.fit(featureMatrix_train, targetMatrix_train)

bestKPredictedLabels = bestKnn_model.predict(featureMatrix_test)

confusionMatrixResult = confusion_matrix(targetMatrix_test, bestKPredictedLabels)
print(f"Confusion Matrix (Best k = {bestK}):\n{confusionMatrixResult}")
print(f"Accuracy: {accuracy_score(targetMatrix_test, bestKPredictedLabels)}")
print(f"Precision: {precision_score(targetMatrix_test, bestKPredictedLabels, average='weighted')}")
print(f"Recall: {recall_score(targetMatrix_test, bestKPredictedLabels, average='weighted')}")
print(f"F1 Score: {f1_score(targetMatrix_test, bestKPredictedLabels, average='weighted')}")


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
