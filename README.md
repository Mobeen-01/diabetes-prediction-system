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
Or manually:
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


