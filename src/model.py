import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

file_path = '../data/train.csv'  
home_data = pd.read_csv(file_path)


home_data = home_data.dropna()

y = home_data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
dt_preds = dt_model.predict(val_X)
dt_mae = mean_absolute_error(val_y, dt_preds)
print(f"Decision Tree MAE: {dt_mae:,.0f}")

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return mae

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y)
          for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)
print(f"Best tree size: {best_tree_size}, MAE: {scores[best_tree_size]:,.0f}")

final_dt_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_dt_model.fit(X, y)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_preds = rf_model.predict(val_X)
rf_mae = mean_absolute_error(val_y, rf_preds)
print(f"Validation MAE for Random Forest Model: {rf_mae:,.0f}")
