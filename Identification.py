import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def classify_iris_csv(csv_filepath):
    """
    Classifies Iris flowers from a CSV file.

    Args:
        csv_filepath (str): Path to the CSV file.

    Returns:
        None (prints model evaluation and visualizations).
    """
    try:
        # Load the Iris dataset from the CSV file
        df = pd.read_csv(csv_filepath)

        # Separate features (X) and target (y)
        X = df.drop('species', axis=1) #Assumes 'species' column exists
        y = df['species']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Random Forest Classifier model
        model = RandomForestClassifier(random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')

        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))

        print('\nConfusion Matrix:')
        print(confusion_matrix(y_test, y_pred))

        # Feature Importance
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        feature_importance_sorted = feature_importance.sort_values(ascending=False)

        print('\nFeature Importance:')
        print(feature_importance_sorted)

        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance_sorted, y=feature_importance_sorted.index)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

        # Visualize Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
    except KeyError:
         print("Error: 'species' column not found in the CSV. Please ensure the CSV has a column named 'species'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage (replace 'iris.csv' with your file path):
classify_iris_csv('IRIS.csv')
