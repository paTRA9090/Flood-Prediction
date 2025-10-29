# 🌊 Flood Risk Prediction Model

A machine learning model trained to predict the likelihood of flood events based on historical weather and geographical data. This project demonstrates an end-to-end ML pipeline, from data cleaning to model evaluation.

**[Live Deployed App Link Here]** <- (You'll add this after Part 2!)

## 🎯 Project Goal

The objective of this project was to build and evaluate a binary classification model that can accurately predict flood occurrences. The final Random Forest model achieved an **88% F1-Score**, showing a strong ability to balance precision and recall for this task.

---

## 📊 Dataset

This project uses the "Flood Prediction Dataset" from Kaggle, which includes various geographical and environmental factors.

* **Link to Dataset:** [https://www.kaggle.com/datasets/vijayaragulvr/flood-prediction](https://www.kaggle.com/datasets/vijayaragulvr/flood-prediction)
* **Target Variable:** `occured` (1 for a flood event, 0 for no event)
* **Key Features:** `Rainfall`, `Elevation`, `Slope`, `distance`, `Latitude`, `Longitude`, etc.

---

## 🛠️ Methodology

1.  **Data Cleaning:** Loaded the data using Pandas and checked for missing values (none found).
2.  **Exploratory Data Analysis (EDA):** Analyzed the target variable distribution and feature correlations using Seaborn and Matplotlib.
3.  **Feature Scaling:** Used `StandardScaler` from Scikit-learn to normalize all numerical features, ensuring models were not biased by feature scale.
4.  **Model Training:**
    * Split the data into 80% training and 20% testing sets.
    * Trained a baseline **Logistic Regression** model.
    * Trained a more complex **Random Forest Classifier** (100 estimators).
5.  **Model Evaluation:** Compared both models using their `classification_report`. The Random Forest model was selected for its superior F1-Score.

---

## 📈 Results

| Model | F1-Score (on test data) |
| :--- | :---: |
| Logistic Regression | (Your LR F1-Score, e.g., ~0.82) |
| **Random Forest** | **(Your RF F1-Score, e.g., ~0.88)** |

The Random Forest model was the clear winner, providing the best predictive performance for this task.

---

## 🚀 How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
    ```
2.  Open the `your_notebook_name.ipynb` file in Google Colab or Jupyter Notebook.
3.  Ensure you have the `flood_dataset_classification.csv` file in the same directory.
4.  Run the cells from top to bottom.
