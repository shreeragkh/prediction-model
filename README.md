# Telecom Customer Churn Prediction Model

## Project Overview
This project aims to predict customer churn for a telecom company using a **Random Forest Classifier**. The dataset contains customer demographics, service usage, billing details, and churn status. By analyzing these factors, the model identifies customers likely to churn, helping the company take proactive retention measures.

## Dataset Description
The dataset (`customer_churn.csv`) includes:
- **Customer Demographics**: Gender, senior citizen status, etc.
- **Service Usage**: Subscription to phone, internet, and security services.
- **Billing & Payments**: Payment method, monthly vs. yearly billing, missed payments.
- **Churn Status**: Whether the customer has churned or not.

## Technologies Used
- **Python**
- **Pandas & NumPy** (Data preprocessing)
- **Matplotlib & Seaborn** (Data visualization)
- **Scikit-learn** (Machine learning)

## Model Development
1. **Data Preprocessing**:
   - Handled missing values by replacing them with `0`.
   - Encoded categorical variables using **one-hot encoding**.
   - Split data into **features (X) and target (y)**.
   - **Train-test split** (70% training, 30% testing).

2. **Model Training**:
   - Used **Random Forest Classifier** for training.
   - Fit the model on the training data.

3. **Model Evaluation**:
   - Predicted churn on the test set.
   - Measured **accuracy**, **confusion matrix**, and **classification report**.
   - Plotted a **confusion matrix heatmap** (`confusion_matrix.png`).

## Installation & Setup
Clone the repository and install dependencies:

```sh
git clone https://github.com/shreeragkh/prediction-model.git
cd <project_directory>
pip install -r requirements.txt

