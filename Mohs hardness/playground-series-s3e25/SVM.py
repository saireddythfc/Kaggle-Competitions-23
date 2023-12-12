import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Read in the data
training = pd.read_csv('playground-series-s3e25/train.csv')
test = pd.read_csv('playground-series-s3e25/test.csv')

# Create a new column called 'target' that is the log of the target variable
#df['target'] = np.log(df['target'])

#Prepare the dataset
train_df = training.drop(['id'], axis=1)


#Normalize the data
train_df=((train_df-train_df.min())/(train_df.max()-train_df.min()) * 10)

#Remove outliers
train_df_no_outliers = train_df[(np.abs(zscore(train_df)) < 3).all(axis=1)]
#print(train_df_no_outliers.shape)

target = train_df_no_outliers.pop('Hardness')
#print(train_df.columns, train_df.shape)
#print("Target", target.shape)
#print(training.columns, training.shape)

test_df = test.drop(['id'], axis=1)
test_df=((test_df-test_df.min())/(test_df.max()-test_df.min()) * 10)
test_df_no_outliers = test_df[(np.abs(zscore(test_df)) < 3).all(axis=1)]
#print(test_df.columns, test_df.shape)

#Dataset Description
#print(train_df.describe())
#print(train_df.info())
#print(train_df.isnull().sum())

missing_data = training.isnull().sum()
#print(missing_data[missing_data > 0])


#Feature selection
correlation_matrix = training.corr().round(4)
correlation_target = abs(correlation_matrix['Hardness'])
relevant_features = correlation_target[correlation_target > 0.15]
relevant_features.drop(['Hardness'], inplace=True)
#print(relevant_features)
plt.figure(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

#print(correlation_matrix.columns, correlation_matrix.shape)

train_df = train_df_no_outliers[relevant_features.index]
#print(train_df.columns, train_df.shape)
#print(target.shape)


#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.3, random_state=42)

#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#Create the regression model
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Accuracy: {}%".format(int(round(accuracy * 100))))
absoluteError = (abs(y_test - y_pred))
medianAbsoluteError = np.median(absoluteError)
#print(absoluteError)
#print(medianAbsoluteError)
print(median_absolute_error(y_test, y_pred))

# plt.plot(y_pred, label='y_pred')
# plt.plot(y_test, label='y_test')
# plt.legend()
# plt.show()


#Predict the values
test_df = test_df[relevant_features.index]
preds = model.predict(test_df)
#print(preds)

#print(max(y_pred), min(y_pred))
#print(max(target), min(target))
#print(max(preds), min(preds))

#Convert the predictions back to normal values
for i, val in enumerate(preds):
    if val < 0:
        preds[i] = 0
    elif val > 10:
        preds[i] = 10
preds = preds.round(3)

#Save the predictions to a CSV file
output = pd.DataFrame({'id': test.id,
                       'Hardness': preds})
output.to_csv('playground-series-s3e25/submission.csv', index=False)
print("Submission was successfully saved!")



