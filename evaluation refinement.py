import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
data = pd.read_csv("user_item_ratings.csv")

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(
    data[['user_id', 'product_id', 'rating']],
    reader
)

trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Initial model
model = SVD(random_state=42)
model.fit(trainset)
predictions = model.test(testset)

# Precision, Recall, F1 calculation
def calculate_metrics(predictions, threshold=4):
    y_true, y_pred = [], []

    for pred in predictions:
        y_true.append(1 if pred.r_ui >= threshold else 0)
        y_pred.append(1 if pred.est >= threshold else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

precision, recall, f1 = calculate_metrics(predictions)

print("Initial Model Performance")
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1-Score :", round(f1, 3))

# Hyperparameter tuning
param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20, 30],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}

grid = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid.fit(dataset)

best_params = grid.best_params['rmse']

# Train refined model
refined_model = SVD(**best_params)
refined_model.fit(trainset)
refined_predictions = refined_model.test(testset)

ref_precision, ref_recall, ref_f1 = calculate_metrics(refined_predictions)

print("\nRefined Model Performance")
print("Precision:", round(ref_precision, 3))
print("Recall   :", round(ref_recall, 3))
print("F1-Score :", round(ref_f1, 3))
