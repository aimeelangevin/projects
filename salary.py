# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Distribution graph function
def distributionGraph(df):
    df.hist('YearsExperience')
    plt.xlabel("Years of Experience")
    plt.ylabel('Count')
    df.hist('Salary')
    plt.xlabel("Salary")
    plt.ylabel('Count')
    plt.show()

# Correlation matrix function
def correlationMatrix(df):
    # Ensure there is enough data to plot correlation matrix
    if (df.shape[1] < 2):
        print("Not enough data for a correlation matrix to be graphed")
        return
    filename = df.dataframeName
    corr = df.corr()
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=10)
    plt.show()

# Scatter matrix function
def scatterMatrix(df):
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction')
    plt.suptitle('Scatter and Density Plot')
    plt.show()

# Define column names
col_list = ["YearsExperience", "Salary"]
# Read data
df1 = pd.read_csv('Salary.csv', delimiter = ',', usecols = col_list)
df1.dataframeName = 'Salary.csv'
# Determine number of rows and columns
numRows, numCols = df1.shape
print(f'There are {numRows} rows and {numCols} columns')

# Visualize the data with the three types of plots
distributionGraph(df1)
correlationMatrix(df1)
scatterMatrix(df1)

years = df1['YearsExperience'].values
salary = df1['Salary'].values
# Split the data into test and train sets
xtrain, xtest, ytrain, ytest = train_test_split(years, salary, train_size = round(numRows*.8), test_size=round(numRows*.2))
xtrain = xtrain.reshape(-1,1)
xtest = xtest.reshape(-1,1)

# Plot the training data
plt.scatter(xtrain, ytrain)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Training Data")
plt.show()

# Use linear regression
model = LinearRegression()
model.fit(xtest, ytest)

x = years
y = salary
# Make predictions
ypredict = model.predict(xtest.reshape(-1,1))
# Plot the predictions
plt.scatter(x, y, color = "blue")
plt.plot(xtest, ypredict, color='red', linewidth=2)
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.title('Experience Years VS Salary')
plt.show()
