# VIVA PREPARATION GUIDE
## Heart Disease Classification Project
**Abdullah Asif (FA23-BCS-017-A) | Abdul Hannan (FA23-BCS-013-A)**

---

## QUICK PROJECT OVERVIEW

**What We Did:**
- Built 5 ML models to predict heart disease (yes/no)
- Dataset: 918 patients, 12 features
- Applied 3 feature engineering techniques (Scaling, Interactions, PCA)
- Compared performance before vs after feature engineering
- Best Model: ANN with F1-score of 0.8789

**Why This Project Matters:**
- Heart disease kills 17.9 million people yearly
- Early detection saves lives
- ML can identify patterns doctors might miss
- Our model catches 87.54% of disease cases with 88.24% precision

---

## SECTION 1: DATASET UNDERSTANDING

### Basic Info
- **Rows:** 918 patients
- **Columns:** 12 features + 1 target
- **Target:** HeartDisease (0=No, 1=Yes)
- **Class Distribution:** 508 disease (55.3%), 410 no disease (44.7%)

### Features Explained

**Numerical Features:**
1. **Age** (28-77): Patient age in years
2. **RestingBP** (0-200): Resting blood pressure in mm Hg
3. **Cholesterol** (0-603): Serum cholesterol in mg/dl (0=missing)
4. **MaxHR** (60-202): Maximum heart rate achieved
5. **Oldpeak** (-2.6 to 6.2): ST depression (heart electrical problem)

**Categorical Features:**
6. **Sex**: M/F
7. **ChestPainType**: ATA, NAP, ASY, TA (4 types)
8. **FastingBS**: 0/1 (blood sugar > 120 mg/dl)
9. **RestingECG**: Normal, ST, LVH (3 types)
10. **ExerciseAngina**: Y/N
11. **ST_Slope**: Up, Flat, Down (3 types)

**Key Findings:**
- ASY chest pain → 80% have disease (counterintuitive!)
- Exercise angina → 75% have disease
- Males have 63% disease rate vs females 25%
- Older age correlates with disease

---

## SECTION 2: DATA PREPROCESSING (STEP-BY-STEP)

### Step 1: Load Data
```python
df = pd.read_csv('heart.csv')
```
**What it does:** Reads CSV file into pandas DataFrame
**Why:** Need data in memory to process it

### Step 2: Handle Missing Values
```python
cholesterol_median = df[df['Cholesterol'] > 0]['Cholesterol'].median()
df_processed.loc[df_processed['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median
```
**What it does:** 
- Line 1: Calculates median cholesterol from non-zero values
- Line 2: Replaces all zero values with median (223 mg/dl)

**Why:** 
- 172 rows (18.7%) had Cholesterol=0 (impossible physiologically)
- These are missing values, not actual zeros
- Median is better than mean (not affected by outliers)

**Viva Question:** *"Why median instead of mean?"*
**Answer:** "Median is robust to outliers. If we had one extreme cholesterol value like 600, mean would be skewed, but median stays stable."

### Step 3: Encode Categorical Variables

**Binary Encoding:**
```python
df_processed['Sex'] = df_processed['Sex'].map({'M': 1, 'F': 0})
df_processed['ExerciseAngina'] = df_processed['ExerciseAngina'].map({'Y': 1, 'N': 0})
```
**What it does:** Converts M/F to 1/0, Y/N to 1/0
**Why:** ML models need numbers, not text. Binary features naturally map to 0/1.

**One-Hot Encoding:**
```python
df_processed = pd.get_dummies(df_processed, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)
```
**What it does:** 
- Creates separate column for each category
- ChestPainType (4 categories) → 3 new binary columns
- `drop_first=True` removes one column to avoid redundancy

**Why:**
- Can't use 0,1,2,3 for ChestPainType (would imply ASY > ATA mathematically)
- Each category needs independent representation
- Drop first to prevent multicollinearity (if 3 columns are 0, we know it's the 4th category)

**Viva Question:** *"Why one-hot for ChestPainType but not for Sex?"*
**Answer:** "Sex has 2 categories (binary), so 0/1 is enough. ChestPainType has 4 categories with no order, so each needs its own column to prevent false ordinal relationships."

### Step 4: Train-Test Split
```python
X = df_processed.drop('HeartDisease', axis=1)
y = df_processed['HeartDisease']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```
**What it does:**
- Line 1-2: Separate features (X) from target (y)
- Line 4: Split 70% train, 30% temporary
- Line 5: Split temporary 50-50 into validation (15%) and test (15%)

**Why:**
- **Train (70%):** Learn patterns
- **Validation (15%):** Tune hyperparameters (not used much here)
- **Test (15%):** Final evaluation on unseen data
- **stratify=y:** Keeps 55:45 ratio in all splits
- **random_state=42:** Makes results reproducible

---

## SECTION 3: FEATURE ENGINEERING (3 TECHNIQUES)

### Technique 1: Feature Scaling (StandardScaler)
```python
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
```
**What it does:**
- `fit_transform(X_train)`: Calculates mean/std from training data AND transforms it
- `transform(X_test)`: Uses training mean/std to transform test data
- Formula: z = (x - mean) / std

**Why:**
- Age ranges 28-77, Cholesterol 0-603 → different scales
- Gradient descent (Logistic Regression, ANN) converges faster with similar scales
- Prevents large features from dominating

**CRITICAL:** Use `fit_transform` on train, but only `transform` on test
**Why:** Prevents data leakage. Test data shouldn't influence training statistics.

**Viva Question:** *"Why not fit_transform on test set?"*
**Answer:** "That would be data leakage. Test set represents future unseen data. We must use training statistics only, otherwise we're 'peeking' at test data during training."

### Technique 2: Interaction Features
```python
X_train_scaled['Age_MaxHR_interaction'] = X_train_scaled['Age'] * X_train_scaled['MaxHR']
X_train_scaled['Oldpeak_MaxHR_interaction'] = X_train_scaled['Oldpeak'] * X_train_scaled['MaxHR']
```
**What it does:** Multiplies two features to create new feature

**Why:**
- **Age × MaxHR:** Captures cardiovascular fitness
  - 60-year-old with MaxHR=180 (good) vs 60-year-old with MaxHR=110 (bad)
  - Linear models can't learn this automatically
- **Oldpeak × MaxHR:** Combines electrical abnormality with heart rate
  - High Oldpeak + Low MaxHR = very concerning

**Domain Knowledge:** From cardiology, we know age and exercise capacity interact

### Technique 3: PCA (Principal Component Analysis)
```python
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```
**What it does:**
- Reduces 17 features → 12 components
- Keeps 95% of variance (information)
- Creates new features that are linear combinations of originals

**How PCA Works (Simple Explanation):**
1. Find direction with most variance in data
2. Project data onto that direction (PC1)
3. Find next direction with most variance (perpendicular to PC1)
4. Repeat until 95% variance captured

**Why:**
- **Dimensionality Reduction:** Fewer features = faster training, less overfitting
- **Noise Removal:** Low variance components often represent noise
- **Uncorrelated Features:** PCA components are orthogonal (independent)

**Results:**
- Original: 17 features
- After PCA: 12 components
- Variance retained: 95.31%

**Viva Question:** *"What is n_components=0.95?"*
**Answer:** "It means keep enough components to preserve 95% of data variance. PCA automatically selects how many components needed. In our case, 12 components captured 95.31% variance."

**Viva Question:** *"Why PCA helps some models but hurts others?"*
**Answer:** "PCA helps gradient-based models (ANN, Logistic Regression) by reducing dimensions and removing noise. But tree-based models prefer original features because splits like 'Age > 55' are interpretable, while 'PC1 > 0.5' is not."

---

## SECTION 4: MODELS EXPLAINED

### Model 1: Logistic Regression

**What It Is:**
- Linear model for binary classification
- Predicts probability using sigmoid function
- Formula: P(disease) = 1 / (1 + e^-(w₁x₁ + w₂x₂ + ... + b))

**Code:**
```python
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
```
**Line-by-line:**
- Line 1: Initialize model with 1000 max iterations
- Line 2: Train on training data (learns weights)
- Line 3: Predict on test data

**Strengths:**
- Fast and interpretable
- Gives probability estimates
- Good baseline

**Weaknesses:**
- Only learns linear boundaries
- Can't capture complex patterns without feature engineering

**Performance:**
- Before FE: F1 = 0.8670
- After FE: F1 = 0.8661 (slight decline)
- Why decline? PCA disrupted linear relationships

### Model 2: Naive Bayes

**What It Is:**
- Probabilistic model using Bayes theorem
- Assumes features are independent (naive assumption)
- Calculates P(disease | features) using P(features | disease)

**Code:**
```python
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred = nb_model.predict(X_test_scaled)
```
**How It Works:**
1. Calculate prior probability: P(disease) from training data
2. For each feature, estimate Gaussian distribution per class
3. For new patient: Calculate likelihood using Gaussian formula
4. Apply Bayes theorem to get posterior probability

**Strengths:**
- Very fast
- Works with small datasets
- Simple and effective

**Weaknesses:**
- Independence assumption often violated (Age and MaxHR are correlated)
- Assumes Gaussian distributions

**Performance:**
- Before FE: F1 = 0.8614
- After FE: F1 = 0.8682 (improved!)
- Why improved? PCA creates independent components, aligning with NB assumption

### Model 3: Decision Tree

**What It Is:**
- Tree structure with decisions at each node
- Splits data based on feature values
- Leaf nodes contain class predictions

**Code:**
```python
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_train)
y_pred = dt_model.predict(X_test_scaled)
```
**Parameters:**
- `max_depth=10`: Limits tree depth (prevents overfitting)

**How It Works:**
1. Start with all data at root
2. Find best feature/value to split (maximize information gain)
3. Create left/right child nodes
4. Recursively repeat for each node
5. Stop at max_depth or when can't split further

**Example Split:**
```
If Oldpeak > 1.5:
    If ST_Slope = Flat:
        Predict: Disease
    Else:
        Predict: No Disease
```

**Strengths:**
- Interpretable (can visualize tree)
- Handles non-linear patterns
- No scaling needed

**Weaknesses:**
- Overfits easily (single tree unstable)
- High variance (small data change → different tree)

**Performance:**
- Before FE: F1 = 0.8227
- After FE: F1 = 0.8205 (declined)
- Why decline? PCA removed interpretable split points

### Model 4: Random Forest

**What It Is:**
- Ensemble of 100 decision trees
- Each tree trained on random data subset
- Final prediction by majority voting

**Code:**
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
```
**Parameters:**
- `n_estimators=100`: Build 100 trees
- `max_depth=10`: Each tree max depth 10

**How It Works:**
1. Create 100 bootstrap samples (random sampling with replacement)
2. For each sample, build a decision tree
   - At each split, only consider √n random features
3. For prediction: Each tree votes, majority wins

**Why Better Than Single Tree:**
- Averaging reduces overfitting
- Random features reduce correlation between trees
- More robust and stable

**Strengths:**
- Best traditional ML performance
- Feature importance rankings
- Robust to overfitting

**Weaknesses:**
- Less interpretable than single tree
- Slower training than simple models

**Performance:**
- Before FE: F1 = 0.8730 (best before)
- After FE: F1 = 0.8719 (slight decline)
- Consistently strong performance

### Model 5: Artificial Neural Network (ANN)

**What It Is:**
- Network of artificial neurons inspired by brain
- Learns through backpropagation
- Our architecture: Input → Hidden(10) → Output(1)

**Code:**
```python
ann_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', 
                          max_iter=500, random_state=42, early_stopping=True)
ann_model.fit(X_train_scaled, y_train)
y_pred = ann_model.predict(X_test_scaled)
```
**Parameters:**
- `hidden_layer_sizes=(10,)`: 1 hidden layer with 10 neurons
- `activation='relu'`: ReLU activation function
- `max_iter=500`: Max 500 training epochs
- `early_stopping=True`: Stop if validation score doesn't improve

**Architecture:**
```
Input Layer: 17 neurons (17 features after encoding + interactions)
    ↓
Hidden Layer: 10 neurons with ReLU activation
    ↓
Output Layer: 1 neuron with Sigmoid activation
```

**How It Works:**
1. **Forward Pass:**
   - Input features multiply by weights
   - Add bias
   - Apply ReLU: f(x) = max(0, x)
   - Hidden outputs multiply by weights
   - Apply Sigmoid: f(x) = 1/(1+e^-x)
   - Get prediction (0 to 1)

2. **Calculate Loss:**
   - Compare prediction to actual label
   - Binary Cross-Entropy loss

3. **Backward Pass (Backpropagation):**
   - Calculate how much each weight contributed to error
   - Use chain rule to propagate error backwards

4. **Update Weights:**
   - Adjust weights using gradient descent
   - Formula: w_new = w_old - learning_rate × gradient

5. **Repeat** until convergence

**Math Example:**
```
Input: [Age=55, MaxHR=150, ...] = x₁, x₂, ...

Hidden Neuron 1:
z₁ = w₁₁×Age + w₁₂×MaxHR + ... + b₁
h₁ = ReLU(z₁) = max(0, z₁)

Output:
z_out = w_out₁×h₁ + w_out₂×h₂ + ... + b_out
prediction = Sigmoid(z_out)
```

**Strengths:**
- Learns complex non-linear patterns
- Flexible architecture
- State-of-the-art capability

**Weaknesses:**
- Black box (hard to interpret)
- Requires more data
- Needs careful hyperparameter tuning

**Performance:**
- Before FE: F1 = 0.8612 (5th place)
- After FE: F1 = 0.8789 (1st place!)
- Why improved? PCA reduced dimensions, helped convergence

---

## SECTION 5: EVALUATION METRICS

### Confusion Matrix
```
                Predicted No    Predicted Yes
Actual No           TN              FP
Actual Yes          FN              TP
```

**Example (ANN After FE):**
```
                Predicted No    Predicted Yes
Actual No           54              8
Actual Yes          10              66
```

### Metrics Formulas & Meaning

**1. Accuracy = (TP + TN) / Total**
- (66 + 54) / 138 = 0.8696 (86.96%)
- Meaning: Overall correctness
- When to use: Balanced classes

**2. Precision = TP / (TP + FP)**
- 66 / (66 + 8) = 0.8824 (88.24%)
- Meaning: Of predicted disease cases, how many actually have disease?
- Clinical: Avoids false alarms

**3. Recall = TP / (TP + FN)**
- 66 / (66 + 10) = 0.8754 (87.54%)
- Meaning: Of actual disease cases, how many did we catch?
- Clinical: Avoids missing disease (critical!)

**4. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)**
- 2 × (0.8824 × 0.8754) / (0.8824 + 0.8754) = 0.8789
- Meaning: Harmonic mean, balances precision and recall
- **Primary metric** for this project

**Viva Question:** *"Why F1-score instead of accuracy?"*
**Answer:** "F1-score balances precision and recall. In medicine, both false positives (unnecessary tests) and false negatives (missed disease) matter. Accuracy can be misleading with imbalanced data. F1 gives us a single number that considers both error types."

---

## SECTION 6: RESULTS SUMMARY

### Before Feature Engineering
| Model | F1-Score | Rank |
|-------|----------|------|
| Random Forest | 0.8730 | 1st |
| Logistic Regression | 0.8670 | 2nd |
| Naive Bayes | 0.8614 | 3rd |
| ANN | 0.8612 | 4th |
| Decision Tree | 0.8227 | 5th |

### After Feature Engineering
| Model | F1-Score | Rank | Change |
|-------|----------|------|--------|
| ANN | 0.8789 | 1st | +0.0177 ⬆️ |
| Random Forest | 0.8719 | 2nd | -0.0011 ⬇️ |
| Naive Bayes | 0.8682 | 3rd | +0.0068 ⬆️ |
| Logistic Regression | 0.8661 | 4th | -0.0009 ⬇️ |
| Decision Tree | 0.8205 | 5th | -0.0022 ⬇️ |

**Key Insight:** Feature engineering helps gradient-based models (ANN, NB) but slightly hurts tree-based models (RF, DT)

---

## SECTION 7: COMPARATIVE ANALYSIS

### Why ANN Won (F1 = 0.8789)

**Reasons:**
1. **Non-linear Learning:** Hidden layer learned complex patterns
2. **PCA Benefit:** Reduced dimensions (17→12) prevented overfitting
3. **Balanced Performance:** High precision (88.24%) AND high recall (87.54%)
4. **Optimization:** Adam optimizer + early stopping worked well

**Clinical Relevance:**
- Catches 87.54% of disease cases (good recall)
- 88.24% precision means only 11.76% false alarms
- Suitable for screening tool

### Why Other Models Lost

**Logistic Regression (F1 = 0.8661):**
- **Linear assumption:** Can't capture non-linear heart disease patterns
- Age effect is non-linear (risk increases exponentially with age)
- Interaction of multiple risk factors hard to model linearly

**Naive Bayes (F1 = 0.8682):**
- **Independence assumption violated:** Age and MaxHR are correlated
- Actually improved with PCA because PC components ARE independent
- Still limited by Gaussian distribution assumption

**Decision Tree (F1 = 0.8205):**
- **Overfitting:** Single tree memorizes training data
- **High variance:** Small changes in data → completely different tree
- **Greedy algorithm:** May find suboptimal splits

**Random Forest (F1 = 0.8719):**
- Actually very strong! Only 0.007 behind ANN
- Slight decline with PCA because trees prefer interpretable features
- More interpretable than ANN (can see feature importance)

### Feature Engineering Impact

**Models That Improved:**
- **ANN (+1.77%):** Dimensionality reduction helped convergence
- **Naive Bayes (+0.68%):** PCA aligned with independence assumption

**Models That Declined (Slightly):**
- **Random Forest (-0.11%):** PCA removed interpretable splits
- **Decision Tree (-0.22%):** Lost ability to split on "Age > 55"
- **Logistic Regression (-0.09%):** PCA disrupted linear relationships

**Lesson:** Feature engineering is model-specific, not universally beneficial

---

## SECTION 8: VISUALIZATION INSIGHTS

### 1. Target Distribution
**Finding:** 508 disease (55.3%), 410 no disease (44.7%)
**Impact:** 
- Class ratio 1.24:1 is acceptable (no SMOTE needed)
- Use F1-score instead of accuracy
- Monitor precision AND recall

### 2. Age Analysis
**Finding:** Disease prevalence increases with age (peak 50-65)
**Impact:**
- Keep Age as continuous (don't bin)
- Age is important predictor
- No age-based filtering needed

### 3. Correlation Heatmap
**Finding:** ST_Slope (0.51), ChestPainType (0.47), ExerciseAngina (0.43) strongest correlations
**Impact:**
- All features useful (none removed)
- Created interaction features based on correlations
- Validated PCA would help

### 4. Categorical Features
**Finding:** ASY chest pain → 80% disease, ExerciseAngina Y → 75% disease
**Impact:**
- One-hot encoding essential (different categories have very different rates)
- Validated domain knowledge
- Guided model expectations

### 5. Numerical Distributions
**Finding:** Cholesterol has 172 zeros, MaxHR shows clear separation
**Impact:**
- Led to median imputation for cholesterol
- Confirmed need for scaling
- Identified strongest numerical predictors

---

## COMMON VIVA QUESTIONS & ANSWERS

### Technical Questions

**Q: Explain PCA in simple terms.**
A: "PCA finds new directions in the data that capture the most variation. Imagine students plotted by study hours and grades. The main direction might be diagonal (students who study more get better grades). PCA finds that diagonal line as PC1. This way, we can describe students with one number instead of two, while keeping most information."

**Q: Why use StandardScaler instead of MinMaxScaler?**
A: "StandardScaler makes mean=0 and std=1. It's better when data has outliers because it doesn't squeeze everything into 0-1 range. MinMaxScaler is sensitive to outliers. For medical data with potential extreme values, StandardScaler is more robust."

**Q: What is overfitting?**
A: "When model memorizes training data instead of learning patterns. Like a student who memorizes answers instead of understanding concepts. Model performs well on training but fails on new data. We prevented this with max_depth limits, early stopping, and PCA."

**Q: Explain backpropagation.**
A: "It's how neural networks learn. Forward: input goes through network to make prediction. Compare prediction to actual answer. Backward: calculate how much each weight contributed to the error. Update weights to reduce error. Repeat until model is accurate."

**Q: Why split into train/validation/test?**
A: "Train: model learns patterns. Validation: check if learning generalizes (not used extensively here). Test: final evaluation on completely unseen data. This prevents overfitting and gives honest performance estimate."

### Project-Specific Questions

**Q: Why did you choose heart disease dataset?**
A: "Heart disease is leading cause of death globally. Early detection saves lives. Dataset has good mix of numerical and categorical features, making it suitable for demonstrating various preprocessing techniques. 918 samples is adequate for traditional ML."

**Q: Which model would you recommend for deployment?**
A: "Depends on priorities. If maximum accuracy needed: ANN (F1=0.8789). If interpretability important: Random Forest (F1=0.8719, only 0.7% behind). If speed critical: Logistic Regression (fast and F1=0.8661). For medical use, I'd choose Random Forest as best balance of accuracy and explainability."

**Q: What are limitations of your project?**
A: "1) Sample size (918) is moderate, deep learning would benefit from more. 2) Missing cholesterol data (18.7%) might introduce bias. 3) Used default hyperparameters; grid search could improve. 4) Single train-test split; k-fold cross-validation would be more robust. 5) No external validation on different hospital data."

**Q: How does your model compare to doctors?**
A: "Our model achieves 87.54% recall (catches 87.54% of cases). This is comparable to screening tests but NOT a replacement for doctors. Model should be used as decision support tool to flag high-risk patients for further examination by cardiologists."

**Q: Explain one-hot encoding with example.**
A: "ChestPainType has 4 values: ATA, NAP, ASY, TA. If we encode as 0,1,2,3, model thinks ASY(3) > TA(2) mathematically, which is wrong. One-hot creates 3 binary columns: ChestPainType_NAP, ChestPainType_ASY, ChestPainType_TA. For ATA patient: [0,0,0]. For ASY patient: [0,1,0]. Now each category is independent."

**Q: Why is recall important in medical diagnosis?**
A: "Recall = TP/(TP+FN) measures how many disease cases we catch. In medicine, missing a disease case (false negative) can be fatal. High recall means we're not missing many patients who actually have disease. Our ANN achieves 87.54% recall, meaning it catches 87.54 out of 100 disease cases."

**Q: What is the difference between Random Forest and Decision Tree?**
A: "Decision Tree: single tree, unstable, overfits easily. Random Forest: ensemble of 100 trees, each trained on random data subset with random features. Final prediction by majority vote. Averaging makes it more robust and accurate. Our results show RF (F1=0.8730) outperforms DT (F1=0.8227)."

**Q: Why did ANN improve so much with PCA while Random Forest didn't?**
A: "ANN uses gradient descent which struggles with high dimensions (curse of dimensionality). PCA reduced 17→12 features, making optimization easier and preventing overfitting. Random Forest uses tree splits which work fine with original features. In fact, RF slightly declined because PCA removed interpretable split points like 'Age > 55'."

---

## KEY NUMBERS TO REMEMBER

**Dataset:**
- 918 patients
- 12 features
- 55.3% disease, 44.7% no disease

**Preprocessing:**
- 172 cholesterol zeros replaced with median (223)
- 70% train, 15% validation, 15% test
- 3 categorical one-hot encoded

**Feature Engineering:**
- 17 features → 12 PCA components
- 95.31% variance retained
- 2 interaction features created

**Best Model (ANN After FE):**
- F1-Score: 0.8789
- Accuracy: 0.8696
- Precision: 0.8824
- Recall: 0.8754
- Improvement: +1.77% over before FE

**Top 3 Features (by correlation):**
1. ST_Slope: 0.51
2. ChestPainType: 0.47
3. ExerciseAngina: 0.43

---

## FINAL TIPS FOR VIVA

1. **Stay Calm:** You know this project inside-out
2. **Be Honest:** If unsure, say "I would need to verify that"
3. **Use Examples:** Always explain with concrete examples
4. **Show Understanding:** Don't just recite, explain WHY
5. **Connect to Domain:** Link ML concepts to medical context
6. **Refer to Results:** Use your actual numbers as proof
7. **Acknowledge Limitations:** Shows critical thinking
8. **Be Ready to Trace Code:** Practice walking through key functions
9. **Understand Trade-offs:** Why one choice over another
10. **Show Enthusiasm:** You built something that could save lives!

**Good luck! You've got this! 🎯**
