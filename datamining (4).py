#!/usr/bin/env python
# coding: utf-8

# In[30]:


# Step 2: Data Collection
import pandas as pd


# In[31]:


# Load data from file or database
data = pd.read_csv("diabetic_data.csv")
data.head()
print(data.columns)


# In[32]:


# Step 3: Data Cleaning
# Remove duplicate rows
import numpy as np



data.isnull().sum()
data.drop_duplicates()
data.dropna(inplace=True)
data.replace('?', np.nan, inplace=True)
data.drop('weight', axis=1, inplace=True)
data.drop('payer_code', axis=1, inplace=True)

data.head()


# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Example: Create a bar plot to compare counts of each gender
plt.figure(figsize=(8, 6))
sns.countplot(x='gender' , data=data)
plt.title('Comparison of which gender is more at risk for having diabetes')
plt.xlabel('gender')
plt.ylabel('number of people with diabetes')
plt.show()

# box plot to see which age at risk
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='age', data=data)
plt.title('')
plt.xlabel('gender')
plt.ylabel('age')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt


category_counts = data['race'].value_counts()

#  a pie plot to see which race is at risk to have diabetes
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Comparison of which race is more at risk for having diabetes')
plt.show()




# In[34]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Select relevant features for clustering
features_for_clustering = [    'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses']

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features_for_clustering])

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
print(kmeans)
# Based on the elbow curve, select the optimal number of clusters and perform clustering
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)


data['Cluster'] = clusters
print(clusters)
# Visualize the clusters using scatter plot (example: using first two features)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data['Cluster'], palette='Set1')
plt.title('Clustering Analysis')
plt.xlabel(features_for_clustering[0])
plt.ylabel(features_for_clustering[1])
plt.legend(title='Cluster')
plt.grid(True)
plt.show()



# In[49]:


from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


selected_columns = [    'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses']


num_samples = 100
sampled_data = data[selected_columns].sample(n=num_samples, random_state=0)

# Perform hierarchical clustering
linked = linkage(sampled_data, method='ward', metric='euclidean')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Perform Agglomerative Clustering
X = data[selected_columns]
k = 3  # Define the number of clusters
agg_cluster = AgglomerativeClustering(n_clusters=k)
clusters = agg_cluster.fit_predict(X)

# Add cluster labels to the dataset
data['Cluster'] = clusters
print(clusters.tostring())


# In[ ]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression
import pandas as pd





data_subset = data[['diag_1', 'diag_2', 'diag_3', 'number_diagnoses']]

# Remove rows with missing values
data_subset = data_subset.dropna()

# Encode non-numeric values
label_encoder = LabelEncoder()
for col in ['diag_1', 'diag_2', 'diag_3']:
    data_subset[col] = label_encoder.fit_transform(data_subset[col])

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_subset.drop('number_diagnoses', axis=1))

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_scaled)
data_subset['kmeans_cluster'] = kmeans.labels_

# K-Means clustering
print("K-Means Clustering:")
print(data_subset.groupby('kmeans_cluster')['number_diagnoses'].mean())
print("-" * 30)




# In[44]:


# Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#Goal: The goal of classification is to predict whether a patient will be readmitted to 
#the hospital within a certain period after being discharged based on 
#various demographic, clinical, and medication-related features.


# Select features and target variable
classification_features = [
    'race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
]
classification_target = 'readmitted'

X = data[classification_features]
y = data[classification_target]

# Preprocessing for categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical features
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Specify average='macro'
recall = recall_score(y_test, y_pred, average='macro')  # Specify average='macro'
f1 = f1_score(y_test, y_pred, average='macro')  # Specify average='macro'

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
print("Classification Report:")
print(class_report)



# In[52]:


# Clustering
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



clustering_features = [
    'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses'
]
X = data[clustering_features]

# Preprocessing for numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = StandardScaler()

# Preprocessing for categorical features (if any)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


X_scaled = preprocessor.fit_transform(X)


k = 5
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(X_scaled)

# cluster centroids in the scaled space
cluster_centroids_scaled = kmeans.cluster_centers_

# Inverse transform the cluster centroids to the original space for numerical features
num_inverse_transformer = preprocessor.named_transformers_['num']
cluster_centroids_numerical = num_inverse_transformer.inverse_transform(cluster_centroids_scaled[:, :len(numerical_features)])

# Inverse transform the cluster centroids to the original space for categorical features (if any)
if len(categorical_features) > 0:
    cat_inverse_transformer = preprocessor.named_transformers_['cat']
    cluster_centroids_categorical = cat_inverse_transformer.inverse_transform(cluster_centroids_scaled[:, len(numerical_features):])
    cluster_centroids = pd.DataFrame(
        data=pd.concat([pd.DataFrame(cluster_centroids_numerical, columns=numerical_features), pd.DataFrame(cluster_centroids_categorical, columns=categorical_features)], axis=1),
        columns=clustering_features
    )
else:
    cluster_centroids = pd.DataFrame(cluster_centroids_numerical, columns=numerical_features)

print(cluster_centroids)
import matplotlib.pyplot as plt

#visulasi
plt.figure(figsize=(8, 6))


feature1 = 'num_lab_procedures'
feature2 = 'num_medications'

# Plot each cluster
for cluster_label in range(k):
    # Filter data points belonging to the current cluster
    cluster_data = X_scaled[cluster_label == cluster_label]
    
    # Plot the data points for the current cluster
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')
#Cluster Centroids: These are the representative values for each feature within each cluster. For numerical features,
#these are the means or medians of the feature values within the cluster. For categorical features, 
#these could be the mode (most frequent category) within the cluster.
# Plot cluster centroids
plt.scatter(cluster_centroids_scaled[:, 0], cluster_centroids_scaled[:, 1], marker='x', color='black', label='Cluster Centroids')

plt.title('Clustering of Patients')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.grid(True)
plt.show()




# In[46]:


# Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


regression_features = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                       'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
                       'number_diagnoses'] 
regression_target = 'readmitted'  # Update with your target variable

X = data[regression_features]
y = data[regression_target]


# Assuming '>30' means readmitted, and converting it to 1
y_numeric = y.apply(lambda x: 1 if x == '>30' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


# In[36]:


import skfuzzy as fuzz
from skfuzzy import control as ctrl

age= ctrl.Antecedent(np.arange(0,101,1),'age')
# Set the membership functions for the output variable
age['Young'] = fuzz.trapmf(age.universe, [18, 18, 27, 44]) # Statistics:min=18, mean=27, max=44
age['Medium'] = fuzz.trapmf(age.universe, [27, 44, 60, 60]) # Customized based␣on provided range
age['Old'] = fuzz.trapmf(age.universe, [44, 44, 60, 60]) # Customized based on␣provided range
age.view()


# In[37]:


num_medications= ctrl.Antecedent(np.arange(0,101,1),'num_medications')
# Set the membership functions for the output variable
num_medications['Low'] = fuzz.trapmf(num_medications.universe, [18, 18, 27, 44]) # Statistics:min=18, mean=27, max=44
num_medications['Medium'] = fuzz.trapmf(num_medications.universe, [27, 44, 60, 60]) # Customized based␣on provided range
num_medications['High'] = fuzz.trapmf(num_medications.universe, [44, 44, 60, 60]) # Customized based on␣provided range
num_medications.view()


# In[38]:


time_in_hospital= ctrl.Consequent(np.arange(0,101,1),'time_in_hospital')
# Set the membership functions for the output variable
time_in_hospital['Low'] = fuzz.trapmf(time_in_hospital.universe, [18, 18, 27, 44]) # Statistics:min=18, mean=27, max=44
time_in_hospital['Medium'] = fuzz.trapmf(time_in_hospital.universe, [27, 44, 60, 60]) # Customized based␣on provided range
time_in_hospital['High'] = fuzz.trapmf(time_in_hospital.universe, [44, 44, 60, 60]) # Customized based on␣provided range
time_in_hospital.view()


# In[39]:


rule1 = ctrl.Rule(age['Young'] & num_medications['Low'], time_in_hospital['Low'])
rule2 = ctrl.Rule(age['Medium'] & num_medications['Medium'],time_in_hospital['Medium'])
rule3 = ctrl.Rule(age['Old'] & num_medications['High'], time_in_hospital['High'])


# In[40]:


#Control system and fuzzy simulator creation
control_system = ctrl.ControlSystem([rule1, rule2, rule3])
fuzzy_simulator = ctrl.ControlSystemSimulation(control_system)


# In[41]:


#Set input values
fuzzy_simulator.input['age'] = 22
fuzzy_simulator.input['num_medications'] = 20

fuzzy_simulator.compute()
time_in_hospital.view(sim=fuzzy_simulator)


# In[42]:


#Set input values
fuzzy_simulator.input['age'] = 40
fuzzy_simulator.input['num_medications'] = 40

fuzzy_simulator.compute()
time_in_hospital.view(sim=fuzzy_simulator)


# In[43]:


#Set input values
fuzzy_simulator.input['age'] = 50
fuzzy_simulator.input['num_medications'] = 50

fuzzy_simulator.compute()
time_in_hospital.view(sim=fuzzy_simulator)


# In[ ]:




