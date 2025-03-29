# Iris Species Classification

This repository contains code for classifying Iris flowers into three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal measurements.

## Dataset

The dataset used is the well-known Iris dataset, which includes the following features:

* **sepal_length:** Length of the sepal (in cm)
* **sepal_width:** Width of the sepal (in cm)
* **petal_length:** Length of the petal (in cm)
* **petal_width:** Width of the petal (in cm)
* **species:** The species of the Iris flower (target variable)

The code can read the data from a `.csv` file.

## Code

The `classify_iris_csv.py` file contains the Python code for building and evaluating the classification model. It uses the following libraries:

* `pandas`: For data manipulation
* `scikit-learn`: For machine learning (Random Forest Classifier, train-test split, evaluation metrics)
* `matplotlib` and `seaborn`: For data visualization

## Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Quantumo0o/Identification-Iris-Flower-Classification
    cd iris-species-classification
    ```

2.  **Install the required libraries:**

    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

3.  **Place your Iris dataset CSV file in the same directory as the `classify_iris_csv.py` file.** Ensure the CSV has columns named 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'.

4.  **Run the script:**

    ```bash
    python classify_iris_csv.py <your_iris_dataset.csv>
    ```

    Replace `<your_iris_dataset.csv>` with the actual name of your CSV file.

## Output

The script will produce detailed output, including:

* **Accuracy: 1.0000** - This indicates a perfect classification result. The model correctly predicted the species of every Iris flower in the test set.
* **Classification Report:**
    * **Precision:** 1.00 for each species. This means that all flowers predicted to belong to a specific species were indeed that species.
    * **Recall:** 1.00 for each species. This shows that the model correctly identified all flowers of each species present in the test set.
    * **F1-score:** 1.00 for each species, representing a perfect balance between precision and recall.
    * **Support:** The number of instances of each species in the test set.
* **Confusion Matrix:**
    * A matrix showing the count of correct and incorrect predictions. In this case, the matrix will have non-zero values only on the diagonal, indicating no misclassifications.
* **Feature Importance:**
    * **petal_length:** The most significant feature for species classification, with an importance score of approximately 0.44.
    * **petal_width:** The second most important feature, with an importance score of approximately 0.42.
    * **sepal_length:** A less significant feature, with an importance score of approximately 0.11.
    * **sepal_width:** The least significant feature, with an importance score of approximately 0.03.
    * This highlights that petal dimensions are far more influential in distinguishing Iris species than sepal dimensions.
* **Visualizations:**
    * A bar plot visualizing the feature importance scores.
    * A heatmap visualizing the confusion matrix, providing a clear visual representation of the model's classification performance.

## Model

The model used is a Random Forest Classifier, which is known for its robustness and excellent performance on classification tasks. The perfect accuracy and perfect classification report metrics indicate the model learned the dataset extremely well.

## Results

The model achieved perfect classification accuracy, indicating that it effectively learned the patterns in the Iris dataset. The petal length and petal width are the most important features for distinguishing between the Iris species, as shown by the feature importance analysis.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
