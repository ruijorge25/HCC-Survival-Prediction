# hcc-survival-prediction

🧠 Supervised Learning: Predicting 1-Year Survival of Patients with HCC

This project applies supervised machine learning techniques to predict the 1-year survival of patients diagnosed with Hepatocellular Carcinoma (HCC), using clinical data provided by the Coimbra Hospital and University Center (CHUC).

📌 Objective
To evaluate and compare the performance of various classifiers and data balancing techniques in predicting patient survivability, with the goal of identifying the most effective combination.

🧪 Classifiers Used

Decision Tree

k-Nearest Neighbors (KNN)

Random Forest

Gradient Boosting

Naive Bayes

AdaBoost

⚖️ Balancing Methods Applied

No balancing

Under-sampling

SMOTE

ADASYN

🔍 Key Findings
The best-performing model was:
• Random Forest with ADASYN

Accuracy: 0.834

Precision: 0.875

Other strong combinations included:
• Gradient Boosting with SMOTE: Accuracy 0.814
• AdaBoost with SMOTE: Accuracy 0.821

These results highlight the importance of data balancing, which significantly improved classifier performance.

📁 Repository Structure

data/
├── hcc_dataset.csv (Raw dataset)
├── modified_file.csv (Cleaned/processed version)
├── hcc_filled.csv (Final version used for training with imputation)

notebooks/
└── Supervised_Learning_Predicting_1_Year_Survival_of_Patients_with_HCC.ipynb

report/
└── Machine_learning_-_HCC.pdf

outputs/
└── hcc_filled.html (HTML table preview of cleaned dataset)

README.md (Project overview and documentation)

🧰 Dependencies

Python 3.x

Jupyter Notebook

pandas, numpy, scikit-learn

imbalanced-learn, matplotlib, seaborn

▶️ How to Run

Clone the repository:
git clone https://github.com/ruijorge25/hcc-survival-prediction.git
cd hcc-survival-prediction

Open the notebook:
jupyter notebook notebooks/Supervised_Learning_Predicting_1_Year_Survival_of_Patients_with_HCC.ipynb

Run the notebook step by step:

Data preprocessing

Model training

Evaluation and results visualization

📊 Methodology

Data Preprocessing
• Handle missing values with imputation
• Encode categorical variables
• Remove outliers and duplicates
• Split into training/testing sets

Model Training
• Apply balancing techniques
• Grid Search for hyperparameter tuning
• Cross-validation to avoid overfitting

Evaluation
• Accuracy and Precision scores
• Confusion Matrix
• ROC curves and bar plots

📎 References

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8727204/

https://www.nature.com/articles/s41598-024-51265-7

https://fortuneonline.org/articles/supervised-machine-learning-techniques-for-the-prediction.pdf

👥 Authors

Beatriz Nogueira Seabra

Pedro Ferreira Oliveira Amaro

Rui Jorge Marques de Almeida
