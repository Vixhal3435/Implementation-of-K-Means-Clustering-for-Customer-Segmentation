![Screenshot 2024-12-17 155515](https://github.com/user-attachments/assets/88031abc-a5e1-4985-a402-8b96a01a2b3e)# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: VISHAL.V
RegisterNumber: 24900179
*/

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv(r"C:\Users\admin\Downloads\Mall_Customers.csv")
print(data.head())
print(data.info())
data.isnull().sum()
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:, 3:])
print(y_pred)
data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 1")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 2")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 3")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 4")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 5")
plt.legend()
plt.title("Customer Segments")
plt.show()

```

## Output:
![Screenshot 2024-12-17 155549](https://github.com/user-attachments/assets/e91ff584-2672-4e0a-b9b5-2bd6fa346b73)

![Screenshot 2024-12-17 155515](https://github.com/user-attachments/assets/3b5ed1a6-ec17-4e86-8f1c-01601e05bd1c)

![Screenshot 2024-12-17 155528](https://github.com/user-attachments/assets/935fcf08-7e3e-459e-ad23-7e6d9aa8d52f)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
