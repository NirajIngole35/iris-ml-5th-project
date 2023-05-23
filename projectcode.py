
#load data
from sklearn.datasets import load_iris
import pandas as pd
df=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\iris ml 5th project\Iris.csv")
print(df)

# summarize data

print(df.info)
print(df.head(5))
print(df.tail(5))
print(df.describe)
print(df.shape)
#data preprocessing
#count instances of each to find missing data

print(df.isnull().sum())#no missing value
print(df['SepalLengthCm'].value_counts())
print(df['SepalWidthCm'].value_counts())
print(df['PetalLengthCm'].value_counts())
print(df['Species'].value_counts())


#find x and y

x=df.drop('Species',axis="columns")
print(x)
y=df.Species
print(y)


#split data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#make a graph 
#visualilze the  all data

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.subplot(3,3,1)
plt.suptitle("Species_Graph")
sns.violinplot(x='Species',y='PetalLengthCm',palette='husl',data=df)
sns.set_style("whitegrid")
plt.title("PetalLengthCm_Graph")
plt.subplot(3,3,2)
sns.violinplot(x='Species',y="SepalWidthCm",palette='husl',data=df)
sns.set_style("whitegrid")
plt.title("SepalWidthCm_Graph")
plt.subplot(3,3,3)
sns.violinplot(x='Species',y='SepalLengthCm',palette='husl',data=df)
sns.set_style("whitegrid")
plt.title("SepalLengthCm _Graph")

#make a  all in one graph 
plt.figure(figsize=(8,4))
sns.pairplot(df.reset_index(), palette="husl", hue="Species", height=3)
#sns.pairplot(x,palette="husl",hue="Species",height=3)
sns.set_style("darkgrid")
plt.show()



# preprocessing StandardScaler
from sklearn .preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)


#model train

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_1=LogisticRegression()
model_1.fit(x_train,y_train)

model_2 = DecisionTreeClassifier()
model_2.fit(x_train, y_train)

model_3 = RandomForestClassifier()
model_3.fit(x_train, y_train)

#predict of model

pred_1=model_1.predict(x_test)
pred_2 = model_2.predict(x_test)
pred_3 =model_3.predict(x_test)

#output
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 # Calculate the accuracy of each model
print("accuracy_score of 1st model:-{0}%".format(accuracy_score(y_test,pred_1)*100))
print("accuracy_score of 2st model:-{0}%".format(accuracy_score(y_test,pred_2)*100))
print("accuracy_score of 3st model:-{0}%".format(accuracy_score(y_test,pred_3)*100))

# Calculate the confusion matrix for each model
print("confusion_matrix of 1st model:-{0}%".format(confusion_matrix(y_test,pred_1)))
print("confusion_matrix of 1st model:-{0}%".format(confusion_matrix(y_test,pred_2)))
print("confusion_matrix of 1st model:-{0}%".format(confusion_matrix(y_test,pred_3)))
