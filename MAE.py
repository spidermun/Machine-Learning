# Kod wczytywania danych ukryty tutaj
import pandas as pd

# Wczytaj dane
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Odfiltruj wiersze z brakującymi wartościami ceny
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Wybierz zmienną docelową i cechy
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor

# Zdefiniuj model
melbourne_model = DecisionTreeRegressor()

# Naucz model
melbourne_model.fit(X, y)