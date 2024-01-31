# This is the code which trains on the whole training set available
# After training, makes prediction on the test set and saves the predicted labels for comparison
# With the Blind test set labels

# Read the data
import pandas as pd
X_train = pd.read_csv("R3_train.csv")
Y_train = pd.read_csv("labels_train.csv")
X_test = pd.read_csv("R3_test.csv")
print('Done!')


from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier

# Check to see if we have any missing data in our features
missing_values = X_train.isnull().sum()
print(missing_values)

missing_values_test = X_test.isnull().sum()
print(missing_values_test)

# Reset indices to ensure alignment
X_train = X_train.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)


# Identify and remove outliers from feature data
z_scores = zscore(X_train)

# Set a threshold for Z-Scores (e.g., 4 standard deviations)
threshold = 4

X_train_no_outliers = X_train[(z_scores < threshold).all(axis=1)]

# Use the index of rows to keep from 'X_train_no_outliers' to subset 'y_train_reset'
Y_train_no_outliers = Y_train.loc[X_train_no_outliers.index]

# Reset indices to ensure alignment
X_train_no_outliers = X_train_no_outliers.reset_index(drop=True)
Y_train_no_outliers = Y_train_no_outliers.reset_index(drop=True)

# Print the number of rows before and after removing outliers
print("Number of rows before removing outliers:", len(X_train))
print("Number of rows after removing outliers:", len(X_train_no_outliers))

# Standard Scaler scaling
X_train = X_train_no_outliers
Y_train = Y_train_no_outliers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Convert the scaled array back to a DataFrame if needed
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Assuming 'X_test' is your DataFrame with outliers
# Scale the test data using the same scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)


from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier

# Doing the PCA on our train and test set
pca = PCA(n_components=96)
X_trans = pca.fit_transform(X_train)
X_train_pca = pd.DataFrame(data= X_trans)
X_test_trans = pca.transform(X_test)
X_test_pca = pd.DataFrame(data = X_test_trans)

# Defining the model
clf = SVC(C= 11, kernel= 'rbf', degree= 3, gamma= 'scale')
bagging_classifier = BaggingClassifier(clf, n_estimators=5, random_state=42)
multioutput_classifier = OneVsRestClassifier(bagging_classifier)

# Train the model on the training set without outliers
multioutput_classifier.fit(X_train_pca, Y_train_no_outliers)

# Make predictions on the test set
predictions = multioutput_classifier.predict(X_test_pca)

# Creating a DataFrame from the predictions
df = pd.DataFrame(data=predictions)

# Save the DataFrame to a CSV file
df.to_csv('Final_labels.csv', index=False)
