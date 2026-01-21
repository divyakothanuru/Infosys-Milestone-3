import pandas as pd

# Load raw data
data = pd.read_csv("raw_data.csv")

# Remove duplicates
data.drop_duplicates(inplace=True)

# Handle missing values
data.dropna(inplace=True)

# Select required columns
data = data[['user_id', 'product_id', 'rating']]

# Save cleaned data
data.to_csv("user_item_ratings.csv", index=False)

print("Data preparation completed successfully.")
