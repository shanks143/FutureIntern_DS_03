# Employee Churn Prediction

## Problem Statement
Build a machine learning model using employee data to predict which employees are likely to leave the company. This model assists organizations in proactively identifying at-risk employees and implementing retention strategies. 

## Features of the Solution
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **Exploratory Data Analysis (EDA)**: Visualizing trends such as average work hours, promotions, salary distributions, and department-wise statistics.
- **Machine Learning Models**: Implemented Logistic Regression and Random Forest Classifier for churn prediction.
- **Hyperparameter Tuning**: Optimized the Random Forest model using GridSearchCV for better accuracy.
- **Evaluation Metrics**: Measured model performance using accuracy, confusion matrix, and classification report (precision, recall, and F1-score).

## Dataset
The dataset used for this project consists of anonymized employee records, including features such as:
- Average monthly hours
- Number of projects
- Time spent in the company
- Work accidents
- Promotions in the last 5 years
- Salary level and department

The dataset can be accessed [here on Kaggle](https://www.kaggle.com/datasets/ksbmishra/employeeturnoverprediction).

---

## Steps Involved

1. **Data Cleaning and Preprocessing**  
   - Removed duplicates and handled missing values.  
   - Encoded categorical features using One-Hot Encoding.  
   - Scaled numerical features using Min-Max Scaling.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized data distributions to understand employee churn factors.  

3. **Model Building and Training**  
   - **Logistic Regression** and **Random Forest** models were implemented and evaluated.  
   - Hyperparameter tuning using GridSearchCV optimized the Random Forest model.

4. **Model Evaluation**  
   - Evaluated model accuracy, precision, recall, F1-score, and confusion matrix.

---

## Results
The Random Forest model achieved the best accuracy after hyperparameter tuning, effectively identifying employees likely to leave the company.

---

## Dependencies
- Python 3.x
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, GridSearchCV, Pickle

---

## Running the Code
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/employee-churn-prediction.git
   cd employee-churn-prediction
