import kagglehub
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
# Download latest version
path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")


# save filepath to variable for easier access
melbourne_file_path = os.path.join(path, 'melb_data.csv')
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
melbourne_data.describe()# print a summary of the data in Melbourne data
print(melbourne_data.columns)
# print(melbourne_data.dropna(axis=0))
# y = melbourne_data.Price
# y.describe()
# y.head()



