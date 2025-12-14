# VIVA PREPARATION (CONCISE BUT COMPLETE)
## Heart Disease Classification Project – DSC293

**Students:** Abdullah Asif (FA23-BCS-017-A), Abdul Hannan (FA23-BCS-013-A)  
**Course:** Data Science Fundamentals  
**Instructor:** Rabail Asghar  
**University:** COMSATS University Islamabad (Lahore Campus)

---

## 1. Project Overview

### What We Did
We developed a **complete end-to-end machine learning classification system** to predict the **presence of heart disease** using real-world clinical data. The project strictly follows the DSC293 final lab requirements and includes preprocessing, feature engineering, visualization, model training, and comparative analysis.

### Why This Problem
Heart disease is the **leading cause of death worldwide**. Early detection helps:
- Identify high-risk patients
- Enable preventive treatment
- Reduce mortality and healthcare costs
- Support doctors with data-driven decision-making

### Workflow
```
Data Loading → EDA & Visualization → Preprocessing →
Feature Engineering → Model Training (5 models) →
Evaluation → Comparison → Best Model Selection
```

### Key Outcome
- **Best Model:** Artificial Neural Network (ANN) after feature engineering
- **Best Metric (F1-score):** **0.8789 (87.89%)**

---

## 2. Dataset Understanding

### Basic Details
- **Dataset:** Heart Disease Dataset (heart.csv)
- **Samples:** 918 patients
- **Features:** 12 input features + 1 target
- **Target Variable:** `HeartDisease`
  - 0 → No disease
  - 1 → Disease present
- **Class Distribution:**
  - Disease ≈ 55%
  - No Disease ≈ 45% (slightly imbalanced but acceptable)

### Important Features (Why They Matter)
- **Age:** Risk increases with age (50–65 most affected)
- **Sex:** Males show higher disease prevalence
- **ChestPainType:** ASY (asymptomatic) shows highest disease rate (silent ischemia)
- **ExerciseAngina:** Strong indicator of reduced blood flow
- **MaxHR:** Lower values indicate poor cardiovascular fitness
- **Oldpeak:** Higher values indicate myocardial ischemia
- **ST_Slope:** Strongest predictor; Flat/Down = high risk

---

## 3. Exploratory Data Analysis (EDA)

We created **5 meaningful visualizations**, each guiding decisions:

1. **Target Distribution (Bar Chart)**
   - Showed slight class imbalance (55:45)
   - Decision: Use **F1-score**, not accuracy alone

2. **Age Distribution (Histogram + Boxplot)**
   - Older patients have higher disease prevalence
   - Decision: Keep Age as a continuous feature

3. **Correlation Heatmap**
   - Strong correlation: ST_Slope, ExerciseAngina, Oldpeak, MaxHR
   - Decision: Create **interaction features** and keep all predictors

4. **Categorical Feature Analysis**
   - ASY chest pain ≈ 80% disease
   - ExerciseAngina = Yes strongly linked to disease
   - Decision: One-hot encoding for nominal categories

5. **Numerical Feature Distributions**
   - Cholesterol had many zero values
   - Decision: Treat zeros as missing and impute

---

## 4. Data Preprocessing

### Missing Value Handling
- **Problem:** Cholesterol had **172 zero values (18.7%)**
- **Reason:** Zero cholesterol is medically impossible → missing data
- **Solution:** Replaced with **median (223 mg/dl)**
- **Why Median:** Robust to outliers, better than mean

### Encoding
- **Binary Encoding:**
  - Sex (M=1, F=0)
  - ExerciseAngina (Y=1, N=0)
- **One-Hot Encoding:**
  - ChestPainType
  - RestingECG
  - ST_Slope
- Used `drop_first=True` to avoid multicollinearity

### Train–Validation–Test Split
- **70% Training / 15% Validation / 15% Test**
- **Stratified split** to preserve class distribution
- **random_state=42** for reproducibility

---

## 5. Feature Engineering (Lecture-16)

### Technique 1: Feature Scaling (StandardScaler)
- Converts features to mean = 0, std = 1
- Required for gradient-based models (Logistic Regression, ANN)
- Prevents dominance of large-scale features

### Technique 2: Interaction Features
- **Age × MaxHR** → captures fitness vs age
- **Oldpeak × MaxHR** → ischemia + exercise capacity
- Helps linear models capture combined effects

### Technique 3: Principal Component Analysis (PCA)
- Reduced features from **17 → 12**
- Preserved **95.31% variance**
- Benefits:
  - Noise reduction
  - Removes multicollinearity
  - Faster training
  - Better generalization

---

## 6. Machine Learning Models Used

1. **Logistic Regression**
   - Linear, interpretable baseline
   - Uses sigmoid function

2. **Naive Bayes (GaussianNB)**
   - Probabilistic model
   - Assumes feature independence

3. **Decision Tree**
   - Rule-based, non-linear model
   - Easy to interpret but prone to overfitting

4. **Random Forest**
   - Ensemble of decision trees
   - Robust, high accuracy, reduced overfitting

5. **Artificial Neural Network (ANN)**
   - One hidden layer (10 neurons, ReLU)
   - Sigmoid output for binary classification
   - Trained using backpropagation and gradient descent

---

## 7. Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** Correct disease predictions
- **Recall:** Ability to catch actual disease cases
- **F1-score:** Balance of precision and recall (**primary metric**)

**Why F1-score?**
In healthcare, both false positives and false negatives are costly.

---

## 8. Results Summary

### Best Results (After Feature Engineering)
- **ANN:**
  - Accuracy: 0.8696
  - Precision: 0.8824
  - Recall: 0.8754
  - **F1-score: 0.8789 (Best Overall)**

### Comparative Insights
- ANN improved the most after feature engineering
- Naive Bayes improved with PCA (orthogonal features)
- Random Forest remained strong and stable
- Decision Tree slightly declined due to PCA
- Logistic Regression stayed a strong baseline

---

## 9. Why ANN Performed Best

- Learns **non-linear relationships**
- Benefited from:
  - Feature scaling
  - PCA-based dimensionality reduction
  - Noise removal
- Early stopping prevented overfitting

---

## 10. Limitations

- Moderate dataset size (918 samples)
- No genetic or lifestyle features
- ANN is a black-box model
- Limited hyperparameter tuning

---

## 11. Conclusion

This project demonstrates a **complete data science workflow**. Feature engineering significantly influenced model performance, and ANN achieved the best balance between precision and recall. The project fully satisfies DSC293 CLOs and demonstrates strong understanding of preprocessing, modeling, evaluation, and comparative analysis.

---

## Viva One-Liner (If Asked)

> “We built a heart disease classification system using five machine learning models. After applying preprocessing and feature engineering including PCA, the Artificial Neural Network achieved the best performance with an F1-score of 87.89%, showing how feature engineering and non-linear models improve medical predictions.”

