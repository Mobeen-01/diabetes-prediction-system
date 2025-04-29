import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.table import Table
from prettytable import PrettyTable
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l1, l2
from keras.layers import LSTM, Dense, Dropout  # Add Dropout import
from keras.regularizers import l1  # Add l1 regularization import
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import AllKNN
from collections import Counter
import pandas as pd
from lightgbm import LGBMClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.callbacks import History
from tensorflow.keras.layers import GRU
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from prettytable import PrettyTable
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.callbacks import History
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np





# Load only model
model = joblib.load("lgbm_model.pkl")

# Validation Ranges
valid_ranges = {
    "Pregnancies": (0, 17),
    "Glucose": (0, 200),
    "BloodPressure": (0, 122),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0, 2.5),
    "Age": (0, 120)
}

# Function to validate and predict

# Updated Predict Function
def predict():
    try:
        # Read input values
        values = {
            "Pregnancies": entry_pregnancies.get(),
            "Glucose": entry_glucose.get(),
            "BloodPressure": entry_bp.get(),
            "SkinThickness": entry_skin.get(),
            "Insulin": entry_insulin.get(),
            "BMI": entry_bmi.get(),
            "DiabetesPedigreeFunction": entry_dpf.get(),
            "Age": entry_age.get()
        }

        # Validation
        for key, val in values.items():
            if val.strip() == "" or val.startswith("Enter"):
                messagebox.showwarning("Missing Input", f"{key} is required.")
                return
            try:
                num = float(val)
                min_val, max_val = valid_ranges[key]
                if not (min_val <= num <= max_val):
                    messagebox.showwarning("Out of Range", f"{key} must be between {min_val} and {max_val}.")
                    return
            except ValueError:
                messagebox.showwarning("Invalid Input", f"{key} must be a number.")
                return

        # Prepare input for prediction
        input_features = np.array([[
            int(values["Pregnancies"]),
            float(values["Glucose"]),
            float(values["BloodPressure"]),
            float(values["SkinThickness"]),
            float(values["Insulin"]),
            float(values["BMI"]),
            float(values["DiabetesPedigreeFunction"]),
            int(values["Age"])
        ]])

        # Predict
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]

        # Display result
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        style = "danger-inverse" if prediction == 1 else "success-inverse"
        result_label.configure(text=f"Prediction: {result}\nProbability: {probability:.2%}", bootstyle=style)

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# Create GUI window
app = ttk.Window(themename="cyborg")
app.title("🩺 PIMA Diabetes Prediction")
app.geometry("670x950")
app.minsize(500, 500)  # Minimum window size

# Heading
ttk.Label(app, text="PIMA Diabetes Predictor", font=("Segoe UI", 24, "bold"), anchor="center").pack(pady=20)

# Input Frame
form_frame = ttk.Frame(app, padding=20)
form_frame.pack(pady=10)

fields = [
    ("Pregnancies (0-17)", "entry_pregnancies"),
    ("Glucose (0-200)", "entry_glucose"),
    ("Blood Pressure (0-122)", "entry_bp"),
    ("Skin Thickness (0-100)", "entry_skin"),
    ("Insulin (0-900)", "entry_insulin"),
    ("BMI (0-70)", "entry_bmi"),
    ("Diabetes Pedigree Function (0-2.5)", "entry_dpf"),
    ("Age (0-120)", "entry_age")
]

entries = {}
for idx, (label_text, var_name) in enumerate(fields):
    ttk.Label(form_frame, text=label_text + ":", font=("Segoe UI", 11), anchor="w").grid(row=idx, column=0, padx=10, pady=5, sticky="w")
    entry = ttk.Entry(form_frame, width=30, font=("Segoe UI", 11), bootstyle="info")
    entry.insert(0, f"Enter {label_text.lower()}")

    def clear_placeholder(event, e=entry):
        if e.get().startswith("Enter"):
            e.delete(0, "end")
            e.configure(foreground="#ffffff" if app.style.theme.name == "cyborg" else "#000000")

    def restore_placeholder(event, e=entry, text=label_text):
        if e.get().strip() == "":
            e.insert(0, f"Enter {text.lower()}")
            e.configure(foreground="#aaa" if app.style.theme.name == "cyborg" else "#333")

    entry.bind("<FocusIn>", clear_placeholder)
    entry.bind("<FocusOut>", restore_placeholder)
    entry.configure(foreground="#aaa" if app.style.theme.name == "cyborg" else "#333")
    entry.grid(row=idx, column=1, padx=10, pady=5, ipady=6, ipadx=4)
    entries[var_name] = entry

# Assign to variables for easy access
entry_pregnancies = entries["entry_pregnancies"]
entry_glucose = entries["entry_glucose"]
entry_bp = entries["entry_bp"]
entry_skin = entries["entry_skin"]
entry_insulin = entries["entry_insulin"]
entry_bmi = entries["entry_bmi"]
entry_dpf = entries["entry_dpf"]
entry_age = entries["entry_age"]

# Predict button
predict_btn = ttk.Button(app, text="🔍 Predict", command=predict, bootstyle="primary-outline", width=20)
predict_btn.pack(pady=20)

# Result display
result_label = ttk.Label(app, text="", font=("Segoe UI", 14, "bold"), padding=20, width=48, anchor="center", bootstyle="info-inverse")
result_label.pack(pady=10)

# Theme switch
def toggle_theme():
    current = app.style.theme.name
    app.style.theme_use("cosmo" if current == "cyborg" else "cyborg")
    for entry in entries.values():
        placeholder_text = entry.get()
        if placeholder_text.startswith("Enter"):
            entry.configure(foreground="#aaa" if app.style.theme.name == "cyborg" else "#333")
        else:
            entry.configure(foreground="#ffffff" if app.style.theme.name == "cyborg" else "#000000")

switch_theme_btn = ttk.Button(app, text="Switch Theme", command=toggle_theme, bootstyle="secondary")
switch_theme_btn.pack(pady=5)

# Footer
ttk.Label(app, text="© 2025 SmartPredictor.ai", font=("Segoe UI", 9), foreground="#888").pack(side="bottom", pady=10)

app.mainloop()
