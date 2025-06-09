# Supervised Learning: Predicting 1-Year Survival of Patients with HCC

This project uses supervised learning techniques to predict the 1-year survival of patients with Hepatocellular Carcinoma (HCC) using various classifiers and data balancing methods.

## Project Description

The objective of this project is to evaluate different machine learning classifiers and data balancing techniques to predict the 1-year survival of HCC patients. The classifiers used include Decision Tree, K-Nearest Neighbors, Random Forest, Gradient Boosting, Naive Bayes, and AdaBoost. The data balancing methods include None, Under-sampling, SMOTE, and ADASYN.

## Prerequisites

- Python 3.x
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - seaborn
  - matplotlib

## Installation

   git clone https://github.com/pedrooamaroo/Supervised-Learning-Predicting-1-Year-Survival-of-Patients-with-HCC.git
   git clone https://github.com/ruijorge25/HCC-Survival-Prediction.git
   git clone https://github.com/beaseabra/HCC-Survival-Prediction.git

Usage.
    1. Open the Jupyter Notebook:
    2. jupyter notebook Supervised_Learning_Predicting_1_Year_Survival_of_Patients_with_HCC.ipynb
    3. Run the notebook cells sequentially to execute the data preprocessing, model training, and evaluation steps.


    data: Contains the dataset used in the project.
    Supervised_Learning_Predicting_1_Year_Survival_of_Patients_with_HCC.ipynb: The main Jupyter Notebook containing the project code.
    README.md: This file.

Methodology
    1. Data Preprocessing:
        ◦ Load the dataset.
        ◦ Handle missing values.
        ◦ Encode categorical variables.
        ◦ Split the data into training and testing sets.
    2. Model Training and Evaluation:
        ◦ Define classifiers and data balancing techniques.
        ◦ Perform grid search for hyperparameter tuning.
        ◦ Train the models with the best parameters.
        ◦ Evaluate the models using precision and accuracy metrics.
        ◦ Visualize the results with confusion matrices and ROC curves.
    3. Results Visualization:
        ◦ Generate bar plots to compare the performance of different classifiers with various balancing techniques.
Results
The results show the performance of different classifiers and data balancing methods in terms of precision and accuracy. The Random Forest classifier with SMOTE balancing achieved the highest accuracy.

