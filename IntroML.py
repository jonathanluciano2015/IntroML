#First, I import essential libraries
import seaborn as sns


from math import exp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

##Kaggle dataset https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset?resource=download
df = pd.read_csv("train.csv")

#After loading the data, I check the first few rows using df.head()  
df.head()
#get summary stats with df.describe(). 
df.describe()

yes_no_columns = ['MultiDeviceAccess', 'PaperlessBilling', 'ParentalControl',
                 'SubtitlesEnabled']

# Convert "yes" to 1 and "no" to 0 for the specified columns
df[yes_no_columns] = df[yes_no_columns].applymap(lambda x: 1 if x == 'yes' else 0)

# Convert 'male' to 1 and 'female' to 0
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

#To ensure a more manageable dataset, I take a random 10% sample.
df = df.sample(frac=0.1
, random_state=42)


#I check for missing values using a heatmap: 
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="Blues")



#The dataset includes categorical features like SubscriptionType and PaymentMethod. 
#use one-hot encoding (pd.get_dummies) to convert these categorical values into numerical

df_subscription_dummies = pd.get_dummies(df['SubscriptionType'],dtype='int')
df_payment_dummies = pd.get_dummies(df['PaymentMethod'],dtype='int')
df_ContentType_dummies = pd.get_dummies(df['ContentType'],dtype='int')
df_DeviceRegistered_dummies = pd.get_dummies(df['DeviceRegistered'],dtype='int')
df_GenrePreference_dummies = pd.get_dummies(df['GenrePreference'],dtype='int')


#concatenate the newly created numerical columns back to the original dataset and 
#drop the categorical columns.
df = pd.concat([df.drop(columns=['SubscriptionType', 'PaymentMethod', 'ContentType','DeviceRegistered','GenrePreference']), 
                   
                      df_subscription_dummies,
                      df_payment_dummies,
                     df_ContentType_dummies,
                      df_DeviceRegistered_dummies,
                      df_GenrePreference_dummies
                     ], axis=1)



# checking for null values
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

#For DF Info, I’m just confirm all datatypes are numbers 
df.info()
#I visualize the distribution of numerical features using histograms.
df.hist(bins = 30, figsize = (20,20), color = 'r')

#Since our target variable is Churn, I analyze the distribution of customers
# stayed vs. those who left: 
left_df        = df[df['Churn'] == 1]
stayed_df      = df[df['Churn'] == 0]



print("Total =", len(df))

print("Number of customers who left =", len(left_df))
print("Percentage of customers who left =", 1.*len(left_df)/len(df)*100.0, "%")
 
print("Number of customers who did not leave (stayed) =", len(stayed_df))
print("Percentage of customers who did not leave (stayed) =", 1.*len(stayed_df)/len(df)*100.0, "%")


left_df.describe()

stayed_df.describe()

df = df.drop('CustomerID', axis=1)
df_important =  df[['AccountAge', 'MonthlyCharges', 'TotalCharges','ViewingHoursPerWeek','AverageViewingDuration','ContentDownloadsPerMonth','UserRating','SupportTicketsPerMonth','WatchlistSize','Churn']]
 
#To understand relationships between features, 
#I compute a correlation matrix and visualize it using a heatmap: 
correlations = df_important.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)

plt.show()


#I also use Kernel Density Estimation (KDE) plots to analyze how certain features 
#(like AccountAge) vary between customers who stayed and those who left. 
plt.figure(figsize=(12,7))

sns.kdeplot(left_df['AccountAge'], label = 'Employees who left', fill = True, color = 'r')
sns.kdeplot(stayed_df['AccountAge'], label = 'Employees who Stayed', fill = True, color = 'b')

plt.xlabel('AccountAge')
plt.show()



plt.figure(figsize=(12,7))

sns.kdeplot(left_df['AverageViewingDuration'], label = 'Employees who left', fill = True, color = 'r')
sns.kdeplot(stayed_df['AverageViewingDuration'], label = 'Employees who Stayed', fill = True, color = 'b')

plt.xlabel('AverageViewingDuration')
plt.show()


 

plt.figure(figsize=(12,7))

sns.kdeplot(left_df['MonthlyCharges'], label = 'Employees who left', fill = True, color = 'r')
sns.kdeplot(stayed_df['MonthlyCharges'], label = 'Employees who Stayed', fill = True, color = 'b')

plt.xlabel('MonthlyCharges')



plt.figure(figsize=(12,7))

sns.kdeplot(left_df['ContentDownloadsPerMonth'], label = 'Employees who left', fill = True, color = 'r')
sns.kdeplot(stayed_df['ContentDownloadsPerMonth'], label = 'Employees who Stayed', fill = True, color = 'b')

plt.xlabel('ContentDownloadsPerMonth')

 

plt.figure(figsize=(12,7))

sns.kdeplot(left_df['UserRating'], label = 'Employees who left', fill = True, color = 'r')
sns.kdeplot(stayed_df['UserRating'], label = 'Employees who Stayed', fill = True, color = 'b')

plt.xlabel('UserRating')
 

plt.figure(figsize=(12,7))

sns.kdeplot(left_df['SupportTicketsPerMonth'], label = 'Employees who left', fill = True, color = 'r')
sns.kdeplot(stayed_df['SupportTicketsPerMonth'], label = 'Employees who Stayed', fill = True, color = 'b')

plt.xlabel('SupportTicketsPerMonth')
#To ensure consistency, I normalize numerical features using MinMaxScaler: 
#X_numerical = df_important[['AccountAge', 'MonthlyCharges', 'TotalCharges','ViewingHoursPerWeek','AverageViewingDuration','ContentDownloadsPerMonth','UserRating','SupportTicketsPerMonth','Gender','WatchlistSize']]
X_numerical = df.drop(columns=['Churn'])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_numerical)

y = df['Churn']
#split the dataset into training and testing sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#I train a logistic regression model to classify whether a customer will churn. 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#I then measure its accuracy and display a confusion matrix. 
from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))
#I also try  Random Forest. 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# I check performance metrics: 
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))
