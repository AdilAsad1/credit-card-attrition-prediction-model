import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize

# read the csv file
df = pd.read_csv('BankChurners.csv')

# replace 'Existing Customer' with 1 and 'Attrited Customer' with 0
df['Attrition_Flag'] = df['Attrition_Flag'].replace({'Existing Customer': 1, 'Attrited Customer': 0})

# replace 'Male' with 1 and 'Female' with 0
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})

#rename columns
df = df.rename(columns={
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'Naive_Bayes_Classifier_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'Naive_Bayes_Classifier_2'
})

df.info()

#check if df contains empty cells
print(df.isnull().sum())

df.head()

df = df.drop('CLIENTNUM', axis=1)

# convert categorical columns to numerical using one-hot encoding
cat_cols = ['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
df = pd.get_dummies(df, columns=cat_cols)

# calculate correlation matrix
corr = df.corr()['Attrition_Flag']
print(corr)

for column in corr.index:
    if corr[column] > 0.5 or corr[column] < -0.5:
        print(column)

corr_matrix = df.corr()
# plot the correlation matrix
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(corr_matrix, cmap=coolwarm, aspect='auto', norm=Normalize(vmin=-1, vmax=1))
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
cbar = fig.colorbar(im)
plt.show()
