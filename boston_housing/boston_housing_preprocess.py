import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras

features = ['crime_capita', 'resindential_land', 'prp_not_retail_bus', 'chas_river', 'nitric_pollution', \
            'avg_rooms', 'age', 'distance', 'rad_highways', 'tax', 'ptratio', 'B-1000', 'LSTAT', 'median_val']

# old version of the dataset
# dataset = pd.read_csv("boston_housing.csv", sep=' ', skipinitialspace=True, names=features)
dataset = pd.read_csv("data/boston_housing.csv", names=features, header=0)
dataset.head(5)

# general information about the dataset
dataset.describe().transpose()

# inspect the correlation matrix between the features and the label
# r in [-1, 1],
# -> where [-1, 0) neg. corr
# -> (0, 1] pos, corr
# -> r = 0 no corr
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True)
plt.show()

# For linear regression, our data should follow a series of assumptions
# According to the heatmap, the feature with the higher correlation is LSTAT (-0.74)
# and average number of rooms (+0.70)
# inspect the linear relationship between the features and the label mesian_val
plt.figure(figsize=(12, 8))
sns.pairplot(data=dataset[['LSTAT', 'avg_rooms', 'median_val']], diag_kind='kde')
plt.show()

#drop the columns with correlation +-0.35
corr = dataset.corrwith(dataset['median_val']).to_numpy()
valid_corr = (lambda x: (x > 0.35) | (x < -0.35))(corr)
dataset = dataset.loc[:, valid_corr]

# we split the data to train and test
train_data = dataset.sample(frac=0.8, random_state=0)
test_data = dataset.drop(train_data.index)

# Write the new data
print("Write data to a file")
train_data.to_csv('data/boston_train_data.csv',index=False)
test_data.to_csv('data/boston_test_data.csv',index=False)
