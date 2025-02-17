import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import time

# Load the cleaned dataset
dataset_path = 'regional_gdp_per_capita_cleaned.csv'
original_df = pd.read_csv(dataset_path)

original_df = original_df.dropna()

region_index = 1
region_name = original_df.loc[region_index, "Region"]
gdp_values = original_df.iloc[region_index, 1:].values

# Normalize the GDP values
scaler = MinMaxScaler()
normalized_gdp_values = scaler.fit_transform(gdp_values.reshape(-1, 1)).flatten()

# Create a comparison DataFrame for validation
comparison_df = pd.DataFrame({
    "Year": original_df.columns[1:].astype(int),
    "Original GDP": gdp_values,
    "Normalized GDP": normalized_gdp_values
})

# Prepare years as input features
years = np.array(comparison_df["Year"]).reshape(-1, 1)
corrected_gdp_values = comparison_df["Normalized GDP"].values

# Normalize the input features
year_scaler = MinMaxScaler()
normalized_years = year_scaler.fit_transform(years)

# BP Neural Network with adjusted parameters
bp_model = MLPRegressor(hidden_layer_sizes=(50, 30), activation='tanh', max_iter=2000, random_state=42)

# Train BP model
start_time_bp = time.time()
bp_model.fit(normalized_years, corrected_gdp_values)
end_time_bp = time.time()

# Predictions with the updated BP Neural Network
bp_predictions = bp_model.predict(normalized_years)


# RBF Neural Network (SVR with RBF kernel)
rbf_model = SVR(kernel='rbf', C=100, epsilon=0.1)

# Train RBF model
start_time_rbf = time.time()
rbf_model.fit(years, corrected_gdp_values)
end_time_rbf = time.time()

# Predictions with RBF Neural Network
rbf_predictions = rbf_model.predict(years)

# Save corrected predictions to an Excel file
corrected_predictions_df = pd.DataFrame({
    "Year": comparison_df["Year"],
    "Original GDP": comparison_df["Original GDP"],
    "Normalized GDP": comparison_df["Normalized GDP"],
    "BP Predictions": bp_predictions,
    "RBF Predictions": rbf_predictions,
})

corrected_excel_file_path = 'corrected_predicted_values.xlsx'
corrected_predictions_df.to_excel(corrected_excel_file_path, index=False)

print(f"Corrected predictions saved to {corrected_excel_file_path}")
