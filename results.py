import pickle

# Load the results from the saved file
with open('model_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Display the comparison
print("\nComparison of BP and RBF Models:")
print(f"BP Model MSE: {results['BP']['MSE']:.4f}")
print(f"RBF Model MSE: {results['RBF']['MSE']:.4f}")
print(f"BP Model Training Time: {results['BP']['Training Time']:.2f} seconds")
print(f"RBF Model Training Time: {results['RBF']['Training Time']:.2f} seconds")
