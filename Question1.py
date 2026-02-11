#Can use
# - scikit-learn(for KNN,train/test split.confusion matrix,metrics)
# - pandas, numpy, matplotlib
#Heads up when I use 'Note:', that is just so I could understand these concepts.

import pandas as pd

#load data frame from csv first
loadCrimeDataFrame = pd.read_csv('crime1.csv')

#Note: focus on one column called 'ViolentCrimesPerPop' in the csv
#Don't do pd.DataFrame that creates a column, instead select column from CSV
neededCrimeData = loadCrimeDataFrame["ViolentCrimesPerPop"]

#First computing for mean
computeMeanFromCrime = neededCrimeData.mean()
print(f"The average violent crimes per population value is: {computeMeanFromCrime}.")
# The average violent crimes per population value is: 0.44119122257053295


#Now compute for median
computeMedianFromCrime = neededCrimeData.median()
print(f"The median for violent crimes per population is: {computeMedianFromCrime}.")
#The median for violent crimes per population is: 0.39


#Now compute for Standard Deviation
computeStandardDeviation = neededCrimeData.std()
print(f"The Standard Deviation for violent crimes per population is: {computeStandardDeviation}.")
#The Standard Deviation for violent crimes per population is: 0.2763505847811399


#Now compute Minimum Value
computeMinimumValue = neededCrimeData.min()
print(f"The minimum value for violent crimes per population is: {computeMinimumValue}.")
#The minimum value for violent crimes per population is :0.02.


#Now compute for Maximum Value
computeMaximumValue = neededCrimeData.max()
print(f"The maximum value for violent crimes per population is: {computeMaximumValue}.")
#The maximum value for violent crimes per population is: 1.0



# - Compare mean and median. Does the distribution look symmetric or skewed?
#Note: mean is sensitive to *outliers* while median represents middle value.
#      -Mean ≈ Median → distribution is approximately symmetric.
#      -Mean > Median → distribution is right-skewed (positively skewed).
#      -Mean < Median → distribution is left-skewed (negatively skewed).

# So, mean(0.44) > median(0.39) this implies it is right skewed(positively).
# This implies there is higher values(outliers) pulling the mean upwards.
# Median remains less affected by these outliers(larger/smaller values).


# - If there are extreme values (very large or very small), which statistic is more affected: mean or median?
# Mean is more affected because it adds all values then divided by # of values.
# This means EVERY data value is included in this total, very large or very small(outliers).
# These affect the final average.
# On the other hand the median is ordered.
# It depends only on positons of the values not the size. Focuses on the middle positions.
# Due to these properties outliers don't affect the median as much. Unless they are in the center of data.
