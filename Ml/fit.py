from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import kagglehub
import os
from sklearn.metrics import mean_absolute_error

# Pobierz dataset
path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")
melbourne_file_path = os.path.join(path, 'melb_data.csv')
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.columns
# Dane z Melbourne mają trochę braków (są domy, dla których niektóre zmienne nie zostały zarejestrowane).
# Obsługę brakujących danych poznamy w kolejnej części kursu.
# Twój zbiór z Iowa nie ma braków w kolumnach, z których korzystasz.
# Na razie wybierzemy najprostsze rozwiązanie i wyrzucimy wiersze z brakami.
# Nie przejmuj się jeszcze tym za bardzo – kod wygląda tak:

# dropna usuwa brakujące wartości (mnemonicznie: na = "not available")
melbourne_data = melbourne_data.dropna(axis=0)


# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()


predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))