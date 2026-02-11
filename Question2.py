import pandas as pd
import matplotlib.pyplot as plt

#load data frame from csv first
loadCrimeDataFrame = pd.read_csv('crime1.csv')

#Note: focus on one column called 'ViolentCrimesPerPop' in the csv
neededCrimeData = loadCrimeDataFrame["ViolentCrimesPerPop"]

#Create Histogram from ViolentCrimesPerPop
#Note: A histogram is a graph that shows the frequency of numerical
#      data using rectangles.(height of 250 people, x-axis will have
#                            the height, y-axis will have number of people.
#                            2 people from 140 to 145cm,53 people from 168 to 173cm
#                            ,45 people from 173 to 178cm,etc.)
plt.figure()
createHistogram = neededCrimeData.plot(kind = 'hist')
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.show()

#What the histogram shows about how the data values are spread? 5-7 sentences.

# This histogram shows how data values are spread from 0 to 1 violent crimes per population.
# Most values are concentrated in the lower to moderate range(0.1-0.5).
# This also shows there is fewer observations appear at the higher end of the x-axis(higher crime->less frequent).
# The shape of this data is somewhat right skewed(longer tail towards higher values->higher frequency(less crime)).
# This data shows that there still is high violent crimes but they still occur less frequent than lower rates of
# violent crimes per population.

#Create Box Plot from ViolentCrimesPerPop
#Note: A box plot displays distribution of data based on minimum,
#      first quartile (Q1), *median* (Q2), third quartile (Q3), and maximum.
plt.figure()
createBoxPlot = neededCrimeData.plot(kind = 'box') # or df'(neededCrimeData)'.boxplot()
plt.title("Box Plot of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Violent Crimes Per Population")
plt.show()

#What the box plot shows about the median? 5-7 sentences.

#Shows median violent crime rate is at about 0.39. This means that half
#of the data falls below 0.39 and half above it. The median is the line
#in the box and is a little bit below the center of overall range.
#This shows that violent crime is moderate rather than being extremely
#high or low implying a balanced tendency. This median is where middle
#of data is mostly concentrated. Median value is the central value of distribution.


#Whether the box plot suggests the presence of outliers? 5-7 sentences.

#The minimum value is about 0.02 and max is about 1 on each end of the handles.
#Q1 is about 0.2, Q2 is about 0.39(median), and Q3 is about 0.65.
#The longer handle above and the few extreme values indicate that
#some communities have unusually high violent crime rates compared
#to rest of the data. The min value is very low but not as extreme to 0.2(Q1)
#so the right skew shows here. Q1 and Q3 is where most of the data is but some
#high values pull it up, confirming potential outliers, this helps explain
#why the mean (0.44) is higher than the median(0.39).