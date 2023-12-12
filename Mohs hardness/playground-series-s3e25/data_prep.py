import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Read in the data
training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create a new column called 'target' that is the log of the target variable
#df['target'] = np.log(df['target'])

#Prepare the dataset
train_df = training.drop(['id'], axis=1)
target = train_df.pop('Hardness')

print(train_df.columns, train_df.shape)
print(target.shape)
#print(training.columns, training.shape)

#Dataset Description
print(training.describe())
print(training.info())
print(training.isnull().sum())

missing_data = training.isnull().sum()
print(missing_data[missing_data > 0])



#Feature selection
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = training.corr()['Hardness']
print(correlation_matrix)
plt.figure(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



#Create the regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Accuracy: {}%".format(int(round(accuracy * 100))))
meadianAbsoluteError = np.median(abs(y_test - y_pred))
print("Median Absolute Error: {} degrees".format(int(round(meadianAbsoluteError))))



#Predict the values
preds = model.predict(test)
print(preds)

#Save the predictions to a CSV file
output = pd.DataFrame({'id': test.id,
                       'Hardness': preds})
output.to_csv('submission.csv', index=False)
print("Submission was successfully saved!")



