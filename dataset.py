#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Define the function to display the summary statistics for a selected column
def summary_statistics(column):
    display(df[[column]].describe())

# Create a dropdown widget for selecting columns
dropdown = widgets.Dropdown(options=list(df.columns), description='Column:')

# Attach the summary_statistics function to the dropdown widget using interact
widgets.interact(summary_statistics, column=dropdown)


# In[2]:


import pandas as pd
from IPython.display import display

# Load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Call the summary_statistics function directly
def summary_statistics():
    display(df.describe())

# Call the summary_statistics function
summary_statistics()


# In[3]:


# Visualization for each parameters


# In[4]:


import plotly.express as px
fig = px.histogram(df, x="Do")
fig.show()


# In[5]:


import plotly.express as px
fig = px.histogram(df, x="Ph")
fig.show()


# In[6]:


import plotly.express as px
fig = px.histogram(df, x="ORP")
fig.show()


# In[7]:


import plotly.express as px
fig = px.histogram(df, x="EC")
fig.show()


# In[8]:


import plotly.express as px
fig = px.histogram(df, x="TDS")
fig.show()


# In[9]:


import plotly.express as px
fig = px.histogram(df, x="Water_Temp")
fig.show()


# In[10]:


import plotly.express as px
fig = px.histogram(df, x="CDO")
fig.show()


# In[11]:


import plotly.express as px
fig = px.histogram(df, x="CpH")
fig.show()


# In[12]:


import plotly.express as px
fig = px.histogram(df, x="CORP")
fig.show()


# In[13]:


import plotly.express as px
fig = px.histogram(df, x="CEC")
fig.show()


# In[14]:


import plotly.express as px
fig = px.histogram(df, x="CTDS")
fig.show()


# In[15]:


import plotly.express as px
fig = px.histogram(df, x="CWT")
fig.show()


# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bqplot.pyplot as plt
from bqplot import Tooltip


# In[17]:


df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")


# In[18]:


df.head()
x = df.values
plt.scatter(x[:,0], x[:,1])


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bqplot.pyplot as plt
from bqplot import Tooltip


# In[ ]:


df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")


# In[ ]:


df.hist(figsize=(10,10))
plt.show()


# In[ ]:


get_ipython().system('pip install mplcursors')


# In[19]:


import plotly.graph_objects as go
import pandas as pd

# Assuming df is your DataFrame containing the correlation data
data = df.corr()

fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='Viridis',  # Use a valid colorscale here
        colorbar=dict(title='Correlation'),
        hovertemplate='x: %{x}<br>y: %{y}<br>Correlation: %{z}<extra></extra>'
    ))

fig.update_layout(
    title='Correlation Heatmap',
    xaxis=dict(title='Features'),
    yaxis=dict(title='Features')
)

fig.show()


# In[20]:


# load and summarize the dataset
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")
df.shape
df
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# summarize the shape of the dataset
print(X.shape, y.shape)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# summarize the shape of the train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[21]:


# evaluate model on the raw dataset
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")
df['wqc'].replace({'Polluted': 5, 'Unpolluted': 6}, inplace=True)
df.shape
df
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# In[22]:


# evaluate model performance with outliers removed using isolation forest
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")
df['wqc'].replace({'Polluted': 5, 'Unpolluted': 6}, inplace=True)
df.shape
df
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# In[23]:


# evaluate model performance with outliers removed using one class SVM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")
df['wqc'].replace({'Polluted': 5, 'Unpolluted': 6}, inplace=True)
df.shape
df
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)


# In[24]:


print(df['wqc'].unique())


# In[25]:


df['wqc'].replace({'Polluted': 5, 'Unpolluted': 6}, inplace=True)


# In[26]:


print(df['wqc'].unique())


# In[27]:


# evaluate model performance with outliers removed using one class SVM
import plotly.express as px
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")
df['wqc'].replace({'Polluted': 5, 'Unpolluted': 6}, inplace=True)
df.shape
df
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
# create a scatter plot of actual vs predicted values
fig = px.scatter(x=y_test, y=yhat, template='plotly_dark', labels={'x': 'Actual', 'y': 'Predicted'})
fig.update_layout(title='Remove outliers using One Class Svm', xaxis_title='Actual', yaxis_title='Predicted')
fig.show()


# In[28]:


import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Define the heatmap function
def plot_heatmap(column):
    plt.figure(figsize=(13,8))
    sns.heatmap(df.corr()[[column]].sort_values(by=column, ascending=False), annot=True, cmap='terrain')
    plt.show()

# Create a dropdown widget for selecting columns
dropdown = widgets.Dropdown(options=list(df.columns), description='Column:')

# Define a function to update the heatmap based on the dropdown value
def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        plot_heatmap(change['new'])

# Attach the on_change function to the dropdown widget
dropdown.observe(on_change)

# Display the dropdown widget
display(dropdown)


# In[29]:


#This code loops over contamination values from 0.01 to 0.1 and for each value, identifies outliers in the training dataset using an Isolation Forest model with the corresponding contamination parameter. It then selects all rows that are not outliers, fits a linear regression model to the cleaned training dataset, and evaluates the model on the test dataset. The resulting MAE value is stored in a list along with the corresponding contamination value. Finally, the code creates a line plot of the MAE values for the different contamination levels using Plotly.


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import plotly.express as px

# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")
df['wqc'].replace({'Polluted': 5, 'Unpolluted': 6}, inplace=True)

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# initialize lists to store contamination and MAE values
contaminations = []
maes = []

# loop over different contamination values
for contamination in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:

    # identify outliers in the training dataset
    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(X_train)
    
    # select all rows that are not outliers
    mask = yhat != -1
    X_train_clean, y_train_clean = X_train[mask, :], y_train[mask]
    
    # fit the model
    model = LinearRegression()
    model.fit(X_train_clean, y_train_clean)
    
    # evaluate the model
    yhat = model.predict(X_test)
    
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    
    # store contamination and MAE values
    contaminations.append(contamination)
    maes.append(mae)

# create a line plot of MAE values for different contamination levels
fig = px.line(x=contaminations, y=maes, labels={'x': 'Contamination', 'y': 'MAE'})
fig.update_layout(title='Model Performance with Outliers Removed')
fig.show()


# In[31]:


import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
import plotly.express as px

# load the dataset
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# summarize the shape of the dataset
print(X.shape, y.shape)

# create a scatter plot to visualize the data
fig = px.scatter(df, x="Do", y="wqc", color="wqc")
fig.show()

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# summarize the shape of the train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Load the dataframe
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Check the contents of the dataframe
print(df.head())

# Check the names of the columns
print(df.columns)

# Create a scatter plot of the first two columns
x = df.values
plt.scatter(x[:,0], x[:,1])
plt.show()


# In[33]:


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Load the dataframe
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Convert categorical variables to numerical values using LabelEncoder
le = LabelEncoder()
df['wqc'] = le.fit_transform(df['wqc'])

# Create input array for IsolationForest
x = df.values

# Fit the IsolationForest model
clf = IsolationForest(contamination=.1)
clf.fit(x)

# Predict outliers using IsolationForest
y_pred = clf.predict(x)


# In[34]:


import numpy as np


# Generate random predictions
predictions = np.random.randn(100)

# Compute the proportion of negative predictions
proportion_negative = (predictions < 0).mean()

# Print the proportion of negative predictions
print(f"Proportion of negative predictions: {proportion_negative:.2f}")


# In[35]:


(predictions<0).mean()


# In[36]:


abn_ind = np.where(predictions < 0)


# In[37]:


pip install mpldatacursor


# In[61]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

# Generate random data
np.random.seed(1)
x = np.random.randn(50, 2)
polluted = np.random.choice([0, 1], size=50, replace=True)

# Generate abnormal indices
abn_ind = np.random.choice(50, 5, replace=False)

# Create a scatter plot with the x and y coordinates
fig, ax = plt.subplots()

# Plot polluted points as red line dots and unpolluted points as blue dots
scatter = ax.scatter(x[:, 0], x[:, 1], c=polluted, cmap='bwr', edgecolors='k')

# Create a custom hover function
def on_hover(sel):
    index = sel.target.index
    x_val, y_val = x[index]
    status = "Polluted" if polluted[index] else "Unpolluted"
    sel.annotation.set_text(f"Status: {status}")

# Connect the hover function to the scatter plot
mplcursors.cursor(hover=True).connect("add", on_hover)

# Show the plot
plt.show()


# In[53]:


import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder

# Load the dataframe
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Convert categorical variables to numerical values using LabelEncoder
le = LabelEncoder()
df['wqc'] = le.fit_transform(df['wqc'])

# Create input array for OneClassSVM
x = df.values

# Fit the OneClassSVM model
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x)

# Predict outliers using OneClassSVM
y_pred = clf.predict(x)

# Add predicted outliers to dataframe
df['outlier'] = np.where(y_pred==-1, True, False)


# In[54]:


import numpy as np


# Generate random predictions
predictions = np.random.randn(100)

# Compute the proportion of negative predictions
proportion_negative = (predictions < 0).mean()

# Print the proportion of negative predictions
print(f"Proportion of negative predictions: {proportion_negative:.2f}")


# In[41]:


abn_ind = np.where(predictions < 0)


# In[42]:


import matplotlib.pyplot as plt
import numpy as np

# Load the dataframe
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Convert categorical variables to numerical values using LabelEncoder
le = LabelEncoder()
df['wqc'] = le.fit_transform(df['wqc'])

# Create input array for One-Class SVM
x = df.values

# Fit the One-Class SVM model
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x)

# Predict outliers using One-Class SVM
y_pred = clf.predict(x)

# Create a scatter plot with the x and y coordinates
plt.scatter(x[:, 0], x[:, 1], c=y_pred)

# Show the plot
plt.show()


# In[43]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM


# Load the dataframe
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Convert categorical variables to numerical values using LabelEncoder
le = LabelEncoder()
df['wqc'] = le.fit_transform(df['wqc'])

# Create input array for One-Class SVM
x = df.values

# Fit the One-Class SVM model
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x)

# Predict outliers using One-Class SVM
y_pred = clf.predict(x)

# Define the color map
colors = np.array(['blue', 'red'])

# Create a scatter plot with the x and y coordinates
plt.scatter(x[:, 0], x[:, 1], c=colors[(y_pred + 1) // 2], cmap=plt.cm.brg)

# Show the plot
plt.show()


# In[44]:


# add visualization code 


# In[50]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM
import mplcursors
get_ipython().system('pip install mplcursors')

# Load the dataframe
df = pd.read_csv(r"E:\SEM 5\PROJECT 1\code\source code 1\july_wqc_edit.csv")

# Convert categorical variables to numerical values using LabelEncoder
le = LabelEncoder()
df['wqc'] = le.fit_transform(df['wqc'])

# Create input array for One-Class SVM
x = df.values

# Fit the One-Class SVM model
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x)

# Predict outliers using One-Class SVM
y_pred = clf.predict(x)

# Define the color map
colors = np.array(['blue', 'red'])

# Create a scatter plot with the x and y coordinates
fig, ax = plt.subplots()
scatter = ax.scatter(x[:, 0], x[:, 1], c=colors[(y_pred + 1) // 2], cmap=plt.cm.brg)

# Set the labels for the tooltip
tooltip_labels = ['Unpolluted' if c == 'blue' else 'Polluted' for c in colors]

# Use mplcursors to show tooltips
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(tooltip_labels[sel.target.index]))

plt.show()


# In[ ]:




