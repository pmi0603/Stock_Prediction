import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from xgboost import XGBClassifier 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import warnings 
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('TSLA.csv') 

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Display basic info
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())

# Plot Tesla Close price with proper dates on the x-axis
plt.figure(figsize=(15,5)) 
plt.plot(df['Date'], df['Close'])  # Use 'Date' as the x-axis
plt.title('Tesla Close Price over Time', fontsize=15) 
plt.xlabel('Date')  # Label x-axis
plt.ylabel('Price in Dollars')  # Label y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)  # Add grid for better visualization
plt.show()

# Remove 'Adj Close' column and check for null values
df = df.drop(['Adj Close'], axis=1)
print(df.isnull().sum())

# Features for visualization
features = ['Open', 'High', 'Low', 'Close', 'Volume'] 

# Distribution plots for each feature
plt.subplots(figsize=(20,10)) 
for i, col in enumerate(features): 
   plt.subplot(2,3,i+1) 
   sb.distplot(df[col]) 
plt.show()

# Boxplots for each feature
plt.subplots(figsize=(20,10)) 
for i, col in enumerate(features): 
   plt.subplot(2,3,i+1) 
   sb.boxplot(df[col]) 
plt.show()

# Create new date-related features
splitted = df['Date'].dt.strftime('%m/%d/%Y').str.split('/', expand=True)
df['day'] = splitted[1].astype('int') 
df['month'] = splitted[0].astype('int') 
df['year'] = splitted[2].astype('int')

# Add a 'quarter end' flag
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0) 
print(df.head())

# Group data by year and visualize
data_grouped = df.groupby('year').mean() 
plt.subplots(figsize=(20,10)) 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']): 
    plt.subplot(2,2,i+1) 
    data_grouped[col].plot.bar() 
plt.show()

# Create new features 'open-close' and 'low-high'
df['open-close'] = df['Open'] - df['Close'] 
df['low-high'] = df['Low'] - df['High'] 

# Create target variable
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Plot the target distribution
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%') 
plt.show()

# Correlation heatmap for features
plt.figure(figsize=(10, 10)) 
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False) 
plt.show() 

# Features and target selection
features = df[['open-close', 'low-high', 'is_quarter_end']] 
target = df['target'] 

# Scaling features
scaler = StandardScaler() 
features = scaler.fit_transform(features) 

# Train-test split
X_train, X_valid, Y_train, Y_valid = train_test_split( 
	features, target, test_size=0.1, random_state=2022) 
print(X_train.shape, X_valid.shape) 

# Model training and evaluation
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()] 

for i in range(3): 
    models[i].fit(X_train, Y_train) 
    print(f'{models[i]} : ') 
    print('Training Accuracy : ', metrics.roc_auc_score( 
        Y_train, models[i].predict_proba(X_train)[:,1])) 
    print('Validation Accuracy : ', metrics.roc_auc_score( 
        Y_valid, models[i].predict_proba(X_valid)[:,1])) 
    print()

# Confusion matrix for the first model
# metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
y_pred = models[0].predict(X_valid)

# Generate the confusion matrix
cm = confusion_matrix(Y_valid, y_pred)

# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
