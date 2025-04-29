# Diabetes Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python)](https://www.python.org)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction

The **Diabetes Prediction System** is a machine learning-based application designed to predict whether an individual is likely to develop diabetes based on various health metrics. The system utilizes a **Pima Indian Diabetes dataset** and applies machine learning models like **LightGBM** and **AllKNN** to accurately predict the probability of diabetes occurrence.

The goal of this system is to assist healthcare professionals in early diabetes diagnosis by automating the prediction process, thereby improving efficiency and decision-making.

## Features

- **Predict Diabetes**: Input health parameters (age, BMI, etc.) and predict the likelihood of developing diabetes.
- **Model Performance Evaluation**: Display model performance metrics like accuracy, precision, recall, F1 score, etc.
- **Graphical User Interface (GUI)**: Simple and intuitive GUI built using **Tkinter** for easy data entry and result display.
- **Report Generation**: Generate performance reports and predictions with ease.


## Benefits

- **Early Detection**: Helps in the early identification of individuals at risk of diabetes, promoting preventive measures.
- **Efficient Prediction**: Provides an automated, accurate prediction based on machine learning models.
- **User-Friendly**: Easy-to-use interface for healthcare professionals to input data and get predictions quickly.
- **Data Integrity**: Uses secure and reliable datasets to ensure accurate predictions.

## Detailed Report 📑

A detailed report outlining the entire project, including methodologies, system execution, and more, has been included in the project folder. You can find the report at:

[Report.pdf](https://github.com/Mobeen-01/diabetes-prediction-system/blob/main/Report.pdf) 📄

This report provides a comprehensive overview of the design, implementation, and functionality of the system.

## Technologies Used

- **Machine Learning**: LightGBM, AllKNN
- **Frontend**: Tkinter (for desktop interface)
- **Backend**: Python
- **Database**: None (uses model predictions directly)
- **Visualization**: Matplotlib, Seaborn


---
## 🧪 Setup & Run the Diabetes Prediction System

Follow the steps below to set up and run the Diabetes Prediction System on your local machine.

### 1. Clone the Repository

First, clone the repository to your local machine by running the following command:

```bash
git clone https://github.com/yourusername/diabetes-prediction-system.git
cd diabetes-prediction-system
```

2. **Install Dependencies**  
Install the required dependencies using requirements.txt:
```bash
pip install -r requirements.txt
```
3. **Install Additional Dependencies**:
   - You also have to install the following dependencies manually:
     ```bash
     pip install imbalanced-learn prettytable xgboost lightgbm ttkbootstrap pandas joblib matplotlib seaborn
     ```
4. **Launch Application**  

Once the dependencies are installed, run the application with the following command:


```bash
python main.py
```

---


## How to Use

### Prerequisites

Before using the Diabetes Prediction System, ensure that you have the following installed:

1. **Python 3.x** for the backend:
   - Install required libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Install Additional Dependencies**:
   -  You also have to install the following dependencies manually:
     ```bash
     pip install imbalanced-learn prettytable xgboost lightgbm ttkbootstrap pandas joblib matplotlib seaborn
     ```






---

## 📁 Project Structure

```
METHOD_2_LGBMCLASSIFIER/
├── data/
│   └── diabetes.csv
├── outputs/
│   ├── accuracy_report.png
│   ├── confusion_matrix.png
│   ├── GUI_Default.png
│   ├── GUI_Diabetes_Detected.png
│   ├── GUI_Non_Diabetes_Detected.png
│   ├── roc_curve.png
│   └── System Workflow Diagram.png
├── lgbm_model.pkl
├── requirements.txt
├── Train.py
├── GUI.py
├── Guide.mp4
├── Report.pdf
```

## 🏗 System Architecture

📌 The project adopts a structured and modular approach, progressing from data preprocessing to model training and final deployment. This architecture ensures maintainability, scalability, and clarity throughout the development lifecycle.

For a visual overview, refer to the [System Workflow Diagram](https://github.com/Mobeen-01/diabetes-prediction-system/blob/main/outputs/System%20Workflow%20Diagram.png) that outlines the entire pipeline.

```

## 📈 Final Model Performance (LightGBM + AllKNN)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 92.6% |
| Precision | 96.1% |
| Recall    | 90.7% |
| F1 Score  | 93.3% |
| ROC-AUC   | 96.0% |





---
## 📜 License

This project is licensed under the MIT License.

---
