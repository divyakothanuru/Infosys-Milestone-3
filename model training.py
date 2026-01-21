import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load prepared data
data = pd.read_csv("user_item_ratings.csv")

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(
    data[['user_id', 'product_id', 'rating']],
    reader
)

# Train-test split
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Train SVD model
model = SVD(random_state=42)
model.fit(trainset)

# Test model
predictions = model.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions)
print("Model training completed.")
