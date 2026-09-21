# ASPER Task 3 — Random Forest Machine Learning Model

## Titanic Survival Prediction

### Objective
Build a Random Forest classification model to predict whether a Titanic passenger survived.

### Dataset
Public Titanic dataset from the Seaborn dataset repository:

- Dataset page: https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv
- Raw CSV: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

### Problem Definition
- **Features:** passenger class, sex, age, family information, fare, embarkation information, and related attributes.
- **Target:** `survived`
  - 0 = did not survive
  - 1 = survived
- **Problem type:** Binary classification.

### Workflow
1. Load dataset
2. Inspect data
3. Remove duplicate records
4. Handle missing values
5. Feature engineering (`family_size`)
6. Exploratory Data Analysis
7. Train/test split
8. Numerical median imputation
9. Categorical most-frequent imputation
10. One-hot encoding
11. Random Forest training
12. Prediction
13. Evaluation using accuracy, precision, recall, F1-score, classification report and confusion matrix

### Model
`RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")`

### Feature Engineering
`family_size = sibsp + parch + 1`

### Reproducibility
Open the notebook in Google Colab and run the cells from top to bottom. The dataset is downloaded automatically. An optional CSV upload cell is also included.

### Results
The notebook calculates the actual evaluation metrics. Run the notebook before submission and use its output. Do not invent metric values.

### Repository Structure
```text
ASPER_Task_3_Random_Forest_Titanic/
├── ASPER_Task_3_Random_Forest_Titanic.ipynb
└── README.md
```
