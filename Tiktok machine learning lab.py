
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")

# Display first few rows
data.head()

# Get number of rows and columns
data.shape

# Get data types of columns
data.dtypes

# Get basic information
data.info()

# Generate basic descriptive stats
data.describe()

# Check for missing values
data.isna().sum()

# Drop rows with missing values
data = data.dropna(axis=0)

# Display first few rows after handling missing values
data.head()

# Check for duplicates
data.duplicated().sum()

# Check class balance
data['claim_status'].value_counts(normalize=True)

# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data['text_length'] = data['video_transcription_text'].str.len()
# Display first few rows of dataframe after adding new column
data.head()

# Calculate the average text_length for claims and opinions.

data[['text_length', 'claim_status']].groupby('claim_status').mean()

# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot
sns.histplot(data=data, stat="count", multiple="dodge", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()

# Select outcome variable
y = data['claim_status']

# Encode target and catgorical variables.
X = data.drop(columns = ['#', 'video_id'])

X['claim_status'] = X['claim_status'].replace({'opinion' : 0, 'claim': 1})

X = pd.get_dummies(X, columns = ['author_ban_status', 'verified_status'], drop_first = True)

# Display first few rows
X.head()

# Assign target variable.
y = X['claim_status']

# Isolate the features.
X = X.drop(['claim_status'], axis = 1)

# Display first few rows of features dataframe
X.head()

# Split data into training and testing sets, 80/20.
X_tr, X_test, y_tr, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0)

# Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size = 0.25, random_state = 0)

# Confirm that the dimensions of the training, validation, and testing sets are in alignment.
X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape

# Fit a random forest model to the training set. Use cross-validation to tune the hyperparameters and select the model that performs best on recall.
X_train = X_train.drop(columns = ['video_transcription_text']).reset_index(drop=True)

# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

get_ipython().run_cell_magic('time', '', 'rf_cv.fit(X_train, y_train)\n')

# Examine best recall score
rf_cv.best_score_

# Examine best parameters
rf_cv.best_params_

# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state = 0)
# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a dictionary of scoring metrics to capture
scoring = {'precision', 'recall', 'f1', 'accuracy'}
# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring = scoring, cv = 5, refit = 'recall')

get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train,y_train)\n')

xgb_cv.best_score_

xgb_cv.best_params_

# Use the random forest "best estimator" model to get predictions on the encoded testing set
X_val = X_val.drop(columns=['video_transcription_text']).reset_index(drop=True)
y_pred = rf_cv.best_estimator_.predict(X_val)

# Display the predictions on the encoded testing set
y_pred

# Display the true labels of the testing set
y_val

# Compute values for confusion matrix
cm = confusion_matrix(y_val,y_pred)
# Create display of confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = None, )
# Plot confusion matrix
disp.plot()
# Display plot
plt.show()

# Create classification report for random forest model
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))

#Evaluate XGBoost model
y_pred = xgb_cv.best_estimator_.predict(X_val)

# Compute values for confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm
                              , display_labels = None)

# Plot confusion matrix
disp.plot()

# Display plot
plt.show()

# Create a classification report
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))

# Use champion model to predict on test data
X_test = X_test.drop(columns=['video_transcription_text']).reset_index(drop=True)
y_pred = rf_cv.best_estimator_.predict(X_test)

# Compute values for confusion matrix
cm = confusion_matrix(y_test,y_pred)

# Create display of confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=None)

# Plot confusion matrix
disp.plot()

# Display plot
plt.show()

# Feature importances of champion model
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()
