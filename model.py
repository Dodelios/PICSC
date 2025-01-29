import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

import preprocess as pp
import warnings
from IPython.display import display

warnings.filterwarnings('ignore')
train_df = pp.train_df.loc[:, :]
test_df = pp.test_df.loc[:, :]
X_train, X_val, y_train, y_val = train_test_split(train_df.drop("Label", axis=1), train_df["Label"], test_size=0.2, random_state=42)
X_test, y_test = test_df.drop("Label", axis=1), test_df["Label"]

# Encode the target variable
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

# Label mapping for the target variable
label_map = {index: label for index, label in enumerate(le.classes_)}

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def train_and_save_model(model, model_name, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    joblib.dump(model, f'{model_name}.joblib')
    val_score = model.score(X_val, y_val)
    return val_score

def train_model(X_train, X_val, y_train, y_val):
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "Extra Trees": ExtraTreesClassifier(),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        "XGBoost": XGBClassifier(),
    }

    scores_list = []

    # Initialize plot for ROC curves
    plt.figure(figsize=(10, 8))

    # Train and evaluate models with progress bar
    for name, model in tqdm(classifiers.items(), desc="Training Models"):
        print(f"Training {name}......")
        
        # Train model and save it
        model.fit(X_train, y_train)
        joblib.dump(model, f'{name}.joblib')
        y_pred = model.predict(X_val)

        # Evaluate predictions for multiclass
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_val, model.predict_proba(X_val), multi_class="ovr")

        cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5))

        # ROC curve for the model (for each class)
        y_proba = model.predict_proba(X_val)
        for i in range(len(np.unique(y_train))):
            fpr, tpr, _ = roc_curve(y_val, y_proba[:, i], pos_label=i)
            plt.plot(fpr, tpr, label=f'{name} - Class {i} (AUC = {roc_auc:.4f})')

        # Append scores to the list
        scores_list.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "CV Score": cv_score
        })

    # Finalize the ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models (Multiclass)')
    plt.legend(loc='lower right')
    plt.show()

    # Create and display the DataFrame of scores
    scores = pd.DataFrame(scores_list)
    return scores

# Run the function
scores = train_model(X_train, X_val, y_train, y_val)
print("Testing Scores between different models:")
print(scores)

# Assuming 'scores' DataFrame has columns 'Model' and 'Accuracy'
models = scores['Model']
accuracy = scores['Accuracy']

# Generate a color map
colors = cm.viridis(np.linspace(0, 1, len(models)))

# Plotting the accuracy points with color for each model
plt.figure(figsize=(10, 6))
for i, (model, acc) in enumerate(zip(models, accuracy)):
    plt.plot(model, acc, marker='o', color=colors[i], markersize=8, label=model)

# Adding a line that passes through the points
plt.plot(models, accuracy, linestyle='-', color='#7393B3', linewidth=2, label="Accuracy Line")

# Customizing the plot
plt.title("Accuracy Scores for Different Models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.98, 1)
plt.xticks(rotation=45)
plt.grid(True)

# Display the plot with legend at the bottom right
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))  # Remove duplicate labels
plt.legend(unique_labels.values(), unique_labels.keys(), loc="lower right")

# Show the plot
plt.tight_layout()
plt.show()

# Save the styled DataFrame to an HTML file for visualization
styled_scores = scores.style.background_gradient(cmap='viridis')
styled_scores.to_html('scores.html')