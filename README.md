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
Then Run manually:
```bash
pip install imbalanced-learn prettytable xgboost lightgbm ttkbootstrap pandas joblib matplotlib seaborn
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








# Diabetes Prediction System

[![Rasa](https://img.shields.io/badge/Rasa-3.x-purple.svg?style=flat&logo=rasa)](https://rasa.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python)](https://www.python.org)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction

The **Diabetes Prediction System** is a machine learning-based application designed to predict the likelihood of a person developing diabetes based on health metrics. It leverages a powerful combination of algorithms, including LightGBM and AllKNN, to provide high-accuracy predictions for both medical professionals and individuals interested in understanding their health risks. The system is built using Python, and the backend uses machine learning models for predictions.

The primary goal of the system is to provide an easy-to-use and accurate tool for diabetes prediction that can be employed in clinical settings or for personal health analysis.

## Features

- **Prediction System**: Predict the likelihood of diabetes using a trained machine learning model.
- **Data Processing**: Includes preprocessing techniques such as imbalanced-learn and SMOTE for better handling of data.
- **Visualization**: Graphical representation of model performance metrics.
- **Model Evaluation**: Performance metrics including accuracy, precision, recall, F1 score, and ROC-AUC.
- **Secure Database**: Store data securely in the database, ensuring no redundancy or missing information.

## Benefits

- **Improved Accuracy**: The system utilizes state-of-the-art machine learning techniques (LightGBM and AllKNN) to provide highly accurate predictions.
- **Efficient and Quick**: Fast and easy processing of health data for diabetes prediction.
- **Data Integrity**: Ensures that all health data is processed accurately with minimal error.
- **User-Friendly Interface**: Simple and intuitive user interface to enter health details and receive predictions.
- **Secure and Private**: All data is processed locally, ensuring no external transmission of sensitive health data.

## Detailed Report 📑

A detailed report outlining the entire project, including methodologies, system execution, and more, has been included in the project folder. You can find the report at:

[Report.pdf](https://github.com/Mobeen-01/diabetes-prediction-system/blob/main/Report.pdf) 📄

This report provides a comprehensive overview of the design, implementation, and functionality of the system.


## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Python
- **Machine Learning Models**: LightGBM, AllKNN
- **Database**: MySQL
- **Web Server**: Flask Container
- **Security**: User Authentication (Login system)

## How to Use

### Prerequisites

Before using the Diabetes Prediction System, ensure that you have the following installed:

1. **Python 3.x** and **Flask** for the backend:
   - Install Flask:
     ```bash
     pip install flask
     ```

2. **MySQL** for the database:
   - Ensure MySQL server is running and accessible.














# 🩺 Diabetes Prediction System using Machine Learning

🎯 An AI-powered tool that predicts diabetes based on medical data with **92.6% accuracy**, using **LightGBM + AllKNN** on the **Pima Indian Diabetes dataset**, integrated into a fully **offline desktop app** via **Tkinter GUI**.

---

## 📊 Overview

This system demonstrates how machine learning can enhance healthcare diagnostics. The final deployed application allows users to enter health metrics and instantly receive predictions — all **offline**, ensuring privacy and accessibility.

---

## 🔧 System Architecture

📌 The project follows a clear, modular workflow from **data preprocessing** to **model deployment**, as illustrated below:

![System Workflow](outputs/System%20Workflow%20Diagram.png)

---

## 🔍 Key Features

✅ 92.6% Accuracy, 96.1% Precision, 90.7% Recall  
✅ Offline-ready — no internet/server dependency  
✅ Built-in GUI using Tkinter with form validation  
✅ Final model: LightGBM + AllKNN Sampling  
✅ ROC-AUC Score: **96%**  
✅ Includes **Dark & Light themes**  
✅ User-friendly display with **real-time prediction**

---

## 💡 How It Works

1. **Input Data**: Users provide values like glucose, BMI, blood pressure, age, etc. via GUI.
2. **Preprocessing**: Data is scaled and transformed as done during training.
3. **Prediction**: The LightGBM model is loaded and predicts diabetes status.
4. **Output**: The result — "Diabetic" or "Non-Diabetic" — along with probability is shown in a color-coded display.

---

## 🧠 Machine Learning Models

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- Feedforward Neural Network (FNN)  
- LightGBM (Final deployed model)

All models were evaluated using:

- Accuracy, Precision, Recall, F1 Score  
- ROC-AUC Curve  
- 5-Fold Cross-Validation  
- GridSearchCV for hyperparameter tuning

---

## 📦 Tech Stack

- Python 3.10  
- Scikit-learn, LightGBM, Imbalanced-learn  
- Tkinter, TTKBootstrap  
- Pandas, NumPy, Matplotlib, Seaborn  
- PrettyTable, Joblib  
- MoviePy (for demo video editing)

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

---

## 🧪 How to Run

1. **Clone the Repository**  
```bash
git clone https://github.com/yourusername/diabetes-prediction-system.git
cd diabetes-prediction-system
```

2. **Install Dependencies**  
```bash
pip install -r requirements.txt
```
Then Run manually:
```bash
pip install imbalanced-learn prettytable xgboost lightgbm ttkbootstrap pandas joblib matplotlib seaborn
```

3. **Launch Application**  
```bash
python main.py
```

---

## 🎥 Demo

▶️ Watch Demo Video  
🖼 See GUI previews in the outputs/ folder.

---

## 📈 Final Model Performance (LightGBM + AllKNN)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 92.6% |
| Precision | 96.1% |
| Recall    | 90.7% |
| F1 Score  | 93.3% |
| ROC-AUC   | 96.0% |

---

## 📚 Dataset

Pima Indian Diabetes Dataset  
📌 Source: UCI Machine Learning Repository  
🔢 768 entries × 8 health features + outcome label

---

## 🛡 Ethical Considerations

- Fully local execution — no data sent externally  
- Model bias addressed using AllKNN + SMOTE  
- Compliant with data privacy standards (GDPR, HIPAA)

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Muhammad Mobeen  
AI/ML Software Engineer  
🔗 LinkedIn


