# IEEE-CIS Fraud Detection — XGBoost პროექტი

## პროექტის მიმოხილვა

ეს პროექტი Kaggle-ის კონკურსის [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) ამოხსნაა. კონკურსის მიზანია საკრედიტო ბარათის ტრანზაქციებში თაღლითობის გამოვლენა. მონაცემები მოწოდებულია Vesta Corporation-ის მიერ და შეიცავს 590,540 სატრენინგო ტრანზაქციას 434 ფიჩერით.

**შეფასების მეტრიკა:** ROC-AUC  
**MLflow ექსპერიმენტები:** [DagsHub](https://dagshub.com/gvakh23/ML_assignment2.mlflow)

---

## რეპოზიტორიის სტრუქტურა

```
├── eda-fraud_detection.ipynb                   # მონაცემების დამუშავება და შეფასება
├── model_experiment_XGBoost.ipynb              # XGBoost ექსპერიმენტები (pipeline ვერსია)
├── model_experiment_LogisticRegression.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_DecisionTree.ipynb
├── model_inference.ipynb                       # საუკეთესო მოდელით პროგნოზი
└── README.md
```

---

## მონაცემების მიმოხილვა

| პარამეტრი | მნიშვნელობა |
|---|---|
| სატრენინგო სტრიქონები | 590,540 |
| ფიჩერები | 434 |
| თაღლითობის წილი | 3.5% (მაღალი დისბალანსი) |
| scale_pos_weight | ~28 |
| ტრენინგის პერიოდი | დღე 0–180 |
| ტესტის პერიოდი | დღე 200–400 |

**მნიშვნელოვანი EDA დასკვნები:**

<img width="1576" height="390" alt="image" src="https://github.com/user-attachments/assets/bce9838d-b0e1-4640-89c0-1c4140557737" />



- **თაღლითობის სიხშირე დროში იცვლება** — 1%-დან 7%-მდე, რაც ნიშნავს რომ მოდელმა დროის მიმართ უნდა განზოგადდეს
- **TransactionAmt ძლიერ გადახრილია** — log-ტრანსფორმაცია სავალდებულოა
- **Train და Test დროით გამოყოფილია** — Train: დღე 0-180, Test: დღე 200-400. გადაფარვა არ არის, ამიტომ random KFold-ის ნაცვლად **TimeSeriesSplit** გამოვიყენეთ


## V Columns NaN Pattern Analysis

<img width="803" height="415" alt="image" src="https://github.com/user-attachments/assets/2edba9c9-36c9-410d-ab3c-dc9991a30d22" />


  
- **V სვეტები (V1-V339)** — 15 განსხვავებული NaN-ის პატერნი. ერთი პატერნის სვეტები ერთი Vesta მოდულიდან მოდის და ძლიერ კორელირებულია ერთმანეთთან

---

## Train/Validation Split

EDA-მ დაადასტურა რომ train და test **დროით გამოყოფილია** — random split-ი გამოიწვევდა data leakage-ს (მომავლის მონაცემები სატრენინგო ნაკრებში მოხვდებოდა). ამიტომ **დროზე დაფუძნებული 80/20 split** გამოვიყენეთ:

```
Train: 472,432 სტრიქონი (დღე 0–145)   fraud rate: 3.51%
Val:   118,108 სტრიქონი (დღე 145–180)  fraud rate: 3.44%
```

Cross-validation-ისთვის `TimeSeriesSplit(n_splits=5)` — ყოველი fold ტრენინგავს წარსულზე და ვალიდაციას აკეთებს მომავალზე.

---

## Preprocessing Pipeline

პრეპროცესინგი დაყოფილია 4 sklearn-compatible კლასად: `Cleaner → CategoricalEncoder → FeatureEngineer → FeatureSelector`. თითოეული `fit()` მხოლოდ სატრენინგო მონაცემებზე ეშვება — val/test-ზე `transform()` მხოლოდ.

### 1. Cleaning

**EDA-ს დასკვნა:** 12 სვეტი >90% null-ით — ვერ შეიცავს სასარგებლო სიგნალს.

<img width="1116" height="512" alt="image" src="https://github.com/user-attachments/assets/1f972ee9-4275-4d26-b754-1c306908c7d5" />

| ნაბიჯი | მიდგომა | მიზეზი |
|---|---|---|
| მაღალი null-ის სვეტები | 90%-ზე მეტი null → წაშლა | >90% null სვეტი ვერ ისწავლის სასარგებლო პატერნს |
| M სვეტები (M1-M9) | T→1, F→0, NaN→fill | Match ფიჩერებია (True/False) — ბინარული ბუნება |
| NaN შევსება | მოდელის მიხედვით განსხვავდება | იხ. ქვემოთ |

**NaN შევსება მოდელის მიხედვით:**

| მოდელი | NaN მეთოდი | მიზეზი |
|---|---|---|
| XGBoost, RF, DT | **-999** | Tree-based — -999 ნორმალური range-ის გარეთაა, ხე ქმნის ცალკე branch-ს missing მნიშვნელობებისთვის, ინახავს missingness სიგნალს |
| Logistic Regression | **mean imputation** | LR linear weighted sum-ს ითვლის — `-999 × w` მასიურ ხელოვნურ სიგნალს შექმნიდა კოეფიციენტებში |

**შედეგი:** 434 → 422 სვეტი

---

### 2. Categorical Encoding

**EDA-ს დასკვნა:** კატეგორიული სვეტები სხვადასხვა კარდინალობით — ერთი მიდგომა ყველასთვის არასწორია.

| სვეტი | მეთოდი | მიზეზი |
|---|---|---|
| `DeviceType` | Binary (desktop→1, mobile→0) | მხოლოდ 2 უნიკალური მნიშვნელობა |
| `ProductCD`, `card4`, `card6` | One-Hot Encoding | <10 უნიკალური — OHE არ ქმნის ზედმეტ სვეტებს, ორდინალური მიმართება არ გულისხმობს |
| `P_emaildomain`, `R_emaildomain`, `DeviceInfo` და სხვ. | Label Encoding | >10 უნიკალური — OHE ასობით sparse სვეტს შექმნიდა |

**მნიშვნელოვანი:** Label Encoder `fit()` მხოლოდ train-ზე. Test-ში უცნობი კატეგორიები → -1.

**შედეგი:** 422 → 435 სვეტი (OHE-მ 13 ახალი სვეტი შექმნა)

---

### 3. Feature Engineering

**EDA-ს დასკვნა:** raw ფიჩერები არ ასახავს ყველა სიგნალს — ახალი ფიჩერების შექმნა საჭიროა.

#### დროის ფიჩერები
```
TransactionDT (raw seconds) → hour, day, month
```
EDA-მ დაადასტურა რომ fraud rate დღის განმავლობაში იცვლება — ღამე უფრო მაღალია.

#### TransactionAmt ტრანსფორმაცია
```
TransactionAmt_log     = log1p(TransactionAmt)   ← skew-ის გასწორება
TransactionAmt_decimal = amount - floor(amount)  ← თაღლითები მრგვალ რიცხვებს იყენებენ
```

#### D სვეტების ნორმალიზაცია
```
D1_norm = D1 - card1_group_min(D1)
```
**EDA-ს დასკვნა:** Train: დღე 0-180, Test: დღე 200-400 — raw D მნიშვნელობები test-ში ~200-ით მეტია. ეს არ ასახავს ქცევის განსხვავებას — მხოლოდ დროის offset-ია. `card1`-ის group minimum-ის გამოკლება ამ drift-ს შლის.

#### UID
```
uid  = card1 + "_" + card2               ← ბარათის ID
uid2 = card1 + "_" + addr1              ← ბარათი + მისამართი  
uid3 = card1 + "_" + addr1 + "_" + D1  ← ინდივიდუალური კლიენტის fingerprint
```
`card1+addr1+D1` ცალკეულ ადამიანს აიდენტიფიცირებს.

#### Email ფიჩერები (top solution-ის მიდგომა)
```
same_email     = (P_emaildomain == R_emaildomain)  ← გამყიდველი/მყიდველი email შეუსაბამობა
P_email_suffix = gmail, yahoo, protonmail... 
```
#### Frequency Encoding
```
card1_freq = card1-ის გამოჩენის სიხშირე train-ში
```
იშვიათი ბარათი/მისამართი = საეჭვო სიგნალი.

#### Aggregation ფიჩერები
```
card1_TransactionAmt_mean/std/max  ← ბარათის ტიპური დანახარჯი
uid_TransactionAmt_mean/std/max    ← კლიენტის ტიპური დანახარჯი
card1_C1_mean, card1_D1_mean...    ← count/time ფიჩერების საშუალო
```
თუ ბარათი ჩვეულებრივ $20-30 ხარჯავს და უეცრად $500-ს ხარჯავს — ეს fraud სიგნალია.

**ყველა stat fit() მხოლოდ train-ზე — val/test-ზე მხოლოდ .map().**

**შედეგი:** 435 → ~506 სვეტი

---

### 4. Feature Selection

| მეთოდი | შედეგი | მიზეზი |
|---|---|---|
| VarianceThreshold(0.01) | რამდენიმე სვეტი წაიშალა | თითქმის მუდმივი სვეტები ვერ განასხვავებს fraud-ს |
| V სვეტების NaN-group შერჩევა | 339 → ~15 V სვეტი | V სვეტები 15 NaN პატერნის ჯგუფად იყოფა. ერთი ჯგუფის სვეტები ძლიერ კორელირებულია — ყველაზე კარგი (isFraud-თან კორელაციით) ვინახავთ |

**V სვეტების შერჩევის ლოგიკა:**
```python
# ჯგუფში 20+ კორელირებული სვეტიდან ვინახავთ მხოლოდ 1-ს
best_col = max(group_cols, key=lambda c: abs(corr(c, isFraud)))
```

**საბოლოო შედეგი:** ~506 → **181 ფიჩერი**

---

## Training ექსპერიმენტები

## Logistic Regression

### Preprocessing განსხვავებები XGBoost-თან შედარებით

Logistic Regression-ი **linear მოდელია** — preprocessing-ი მნიშვნელოვნად განსხვავდება tree-based მოდელებისგან:

| ნაბიჯი | XGBoost | Logistic Regression | მიზეზი |
|---|---|---|---|
| NaN შევსება | -999 | mean imputation | LR linear weighted sum-ს ითვლის — -999 კოეფიციენტებს მასიურად გაამახინჯებდა |
| Scaling | ❌ | ✅ StandardScaler | LR gradient descent-ს იყენებს — სხვადასხვა scale-ის ფიჩერები კონვერგენციას ანელებს |
| Correlation filter | ❌ | ✅ (>0.95) | multicollinearity LR კოეფიციენტებს არასტაბილურს ხდის — tree-ებისთვის ეს პრობლემა არ არის |
| class_weight | scale_pos_weight=28 | `balanced` | LR-ს scale_pos_weight-ის ეკვივალენტი |
| Sampling for speed | ❌ | ✅ 20% subsample | 590k სტრიქონზე SAGA solver-ი საათობით მუშაობს |

**რატომ mean imputation LR-ისთვის:**
-999 ასე მუშაობს LR-ში: `w × (-999)` = კოეფიციენტი × (-999) = ძალიან დიდი უარყოფითი რიცხვი logit-ში. ეს ყველა missing feature-ს ხელოვნურ fraud სიგნალად (ან anti-fraud სიგნალად) აქცევს. mean imputation კი neutral-ია — missing მნიშვნელობა საშუალოს იღებს და logit-ს არ ამახინჯებს.

**რატომ StandardScaler სავალდებულოა:**
TransactionAmt range: 0-30,000. V ფიჩერები range: 0-1. scaling-ის გარეშე gradient: `∂L/∂w_amount >> ∂L/∂w_V` — amount ფიჩერი დომინირებს და სხვა ფიჩერები ვერ სწავლობს.

---

### Training ექსპერიმენტები

| Run სახელი | val AUC | train AUC | Gap | შენიშვნა |
|---|---|---|---|---|
| `LR_Underfitted` | 0.740 | 0.744 | 0.004 | C=0.001 — ძლიერი regularization |
| `LR_Overfitted` | 0.747 | 0.760 | 0.013 | C=1000 — regularization-ის გარეშე |
| `LR_Baseline` | 0.744 | 0.749 | 0.005 | C=1.0, saga solver |
| `LR_Tuned` ⭐ | **0.811** | 0.843 | 0.032 | C=0.05, L2, 20% sample |

---

**LR-ის შეზღუდვები ამ dataset-ზე:**

**მთავარი დასკვნა:** `LR_Underfitted` (val: 0.740) და `LR_Overfitted` (val: 0.747) — **თითქმის ერთნაირია!** regularization-ის ცვლილება performance-ს მნიშვნელოვნად არ ცვლის, რადგან პრობლემა overfitting კი არ არის, არამედ **model capacity** 

---

## Decision Tree

პრეპროცესინგის განსხვავება:
1. `balanced` ან undersampling გამოიყენება მონაცემების დასაბალანსებლად.
ორივე გავტესტეთ (იხ. ქვემოთ)
2. scaling არ სჭირდება რადგან ხის მოდელია.

---

### Training ექსპერიმენტები

| Run სახელი | val AUC | train AUC | Gap | შენიშვნა |
|---|---|---|---|---|
| `DT_Underfitted` | 0.767 | 0.780 | 0.013 | max_depth=3 — ძალიან მარტივი |
| `DT_Overfitted` | 0.822 | 1.000 | 0.178 | max_depth=None — სრული memorization |
| `DT_Baseline` | 0.836 | 0.925 | 0.089 | სტანდარტული პარამეტრები |
| `DT_Tuned` | 0.845 | 0.904 | 0.059 | regularization გაძლიერდა, CV mean: 0.833 |
| `DT_NoClassWeight` | 0.812 | 0.838 | 0.026 | class weighting გარეშე |
| `DT_undersampling` | 0.823 | 0.882 | 0.059 | RandomUnderSampler(0.1) |
| `DT_balancing` | 0.832 | 0.904 | 0.072 | class_weight='balanced', full data |
| `DT_Tuned_us` ⭐ | **0.835** | 0.881 | 0.047 | undersampling + max_features=0.5 |

---

### ანალიზი

**Underfitted vs Overfitted:**
`DT_Underfitted` (val: 0.767) — max_depth=3-თ, ასეთი მარტივი მოდელი ვერ ისწავლის fraud-ის კომპლექსურ პატერნებს — train AUC-იც (0.780) დაბალია. `DT_Overfitted` (val: 0.822, train: 1.000, gap: 0.178) — შეუზღუდავი სიღრმე ხეს ყოველ სატრენინგო მაგალითამდე ზრდის. gap 0.178 — ყველა run-ს შორის ყველაზე მაღალი, კლასიკური overfitting.

**Class Weighting vs Undersampling:**
`DT_NoClassWeight` (val: 0.812) — weighting-ის გარეშე მოდელი 96.5% non-fraud მაგალითებით ისწავლის და fraud-ს უგულებელყოფს. `DT_balancing` (class_weight='balanced', val: 0.832) vs `DT_undersampling` (val: 0.823) — balanced weighting სჯობს undersampling-ს.

**საუკეთესო კომბინაცია — `DT_Tuned_us`:**
max_depth=8, min_samples_leaf=50, min_samples_split=80, max_features=0.5, undersampling. gap: 0.047 — ყველა run-ს შორის ყველაზე დაბალი, ყველაზე კარგი გენერალიზაცია. max_features=0.5 ნიშნავს ყოველ split-ზე 50% ფიჩერის განხილვა — ამას დამატებით რენდომიზაცია შემოაქვს და overfitting-ს ამცირებს.

DT **ერთი ხეა** — ის ვერ ისარგებლებს ensemble-ის სიძლიერით. XGBoost ათასობით ხეს აწყობს თანმიმდევრულად და ყოველი შემდეგი წინისგან სწავლობს. DT-ი კი ერთი გადაწყვეტილებას იღებს, რაც fraud detection-ის მსგავს კომპლექსურ, high-dimensional პრობლემაზე ეს სტრუქტურული შეზღუდვაა.

---

## RandomForest_training ექსპერიმენტი

### პრეპროცესინგის განსხვავება:
1. scaling არ სჭირდება რადგან ხის მოდელია.
2. **`class_weight='balanced'` undersampling-ის ნაცვლად RF-ში:**
RF აწყობს ბევრ ხეს **დამოუკიდებლად** — ყოველ ხეს სჭირდება საკმარისი მონაცემი სწავლისთვის. Undersampling 590k → 180k ამცირებს მონაცემებს, ვინაიდან, RF ყოველი ხისთვის bootstrap sample-ს იყენებს, ამ უკვე შემცირებული მონაცემებიდან — ძალიან ცოტა რჩება და შედეგი უარესდება. `class_weight='balanced'` კი სრულ მონაცემებს ინახავს და fraud class-ს უბრალოდ მეტ წონას ანიჭებს.

---

### Training რამდენიმე მნიშვნელოვანი ექსპერიმენტი

| Run სახელი | val AUC | train AUC | Gap | შენიშვნა |
|---|---|---|---|---|
| `RF_Underfitted` | 0.851 | 0.874 | 0.023 | max_depth=3, 10 ხე |
| `RF_Overfitted` | 0.908 | 1.000 | 0.092 | max_depth=None, min_samples_leaf=1 |
| `RF_Baseline_Undersampled` | 0.897 | 0.964 | 0.067 | 200 ხე, sqrt features |
| `RF_MoreTrees_US` | 0.897 | 0.964 | 0.068 | 500 ხე — diminishing returns |
| `RF_Tuned_US` | 0.899 | 0.972 | 0.073 | max_depth=20, min_leaf=20 |
| `RF_MaxFeatures_30pct_US` | 0.898 | 0.972 | 0.074 | max_features=0.3 |
| `RF_MaxFeatures_30pct_Full` ⭐ | **0.912** | 0.996 | 0.083 | full data + class_weight=balanced |

---

**Underfitted vs Overfitted:**
`RF_Underfitted` (val: 0.851) — max_depth=3 და 10 ხე ნიშნავს რომ მოდელი მხოლოდ ყველაზე ზედაპირულ პატერნებს ისწავლის. train AUC-იც (0.874) დაბალია — კლასიკური underfitting. `RF_Overfitted` (val: 0.908, train: 1.000, gap: 0.092) — შეუზღუდავი სიღრმის ხეები ყველა სატრენინგო მაგალითს იმახსოვრებს, val-ზე კი ვერ განზოგადდება.

**Undersampling vs Full Data:**
Undersampled run-ები (0.897-0.899) სტაბილურად უარესია ვიდრე full data (0.912). ეს ადასტურებს რომ RF-ისთვის `class_weight='balanced'` სჯობია — მონაცემების დაკარგვა RF-ს უფრო მეტად აზარალებს ვიდრე XGBoost-ს, რადგან RF-ი მონაცემების სიმრავლეზეა დამოკიდებული.

**საუკეთესო კომბინაცია:**
`RF_MaxFeatures_30pct_Full` — სრული სატრენინგო მონაცემები + `max_features=0.3` + `class_weight='balanced'`. max_features=0.3 ნიშნავს რომ ყოველ split-ზე 181-დან ~54 ფიჩერი განიხილება — ეს ქმნის უფრო მრავალფეროვან ხეებს sqrt-თან (~13 ფიჩერი) ან 0.3-თან შედარებით.

RF ყველაზე ახლოს მივიდა XGBoost-თან ყველა სხვა მოდელს შორის, მაგრამ ვერ გადააჭარბა. ეს მოსალოდნელია — RF აწყობს ხეებს **დამოუკიდებლად** (bagging), XGBoost კი **თანმიმდევრულად** (boosting) — ყოველი ახალი ხე წინა ხის შეცდომებს ასწორებს. fraud detection-ის მსგავს კომპლექსურ, დაუბალანსებელ პრობლემაზე sequential learning-ი მკაფიოდ უჯობს parallel learning-ს.

---

### XGBoost_Training ექსპერიმენტი
მეტრიკა: **ROC-AUC**

#### ყველა run-ის შედეგები

| Run სახელი | val AUC | train AUC | Gap | შენიშვნა |
|---|---|---|---|---|
| `XGBoost_Underfitted` | 0.867 | 0.891 | 0.024 | max_depth=2, 50 ხე — ძალიან მარტივი |
| `XGBoost_Overfitted` | 0.905 | 1.000 | 0.095 | max_depth=15, no regularization |
| `XGBoost_NoFE` | 0.913 | 0.978 | 0.065 | Feature Engineering გარეშე |
| `XGBoost_Baseline` | 0.919 | 0.984 | 0.064 | სტანდარტული პარამეტრები |
| `XGBoost_Tuned` | 0.919 | 0.975 | 0.055 | regularization გაძლიერდა |
| `XGBoost_EarlyStopping` | 0.922 | 0.984 | 0.062 | 3772 ოპტიმალური ხე |
| `XGBoost_Undersampling` | 0.921 | 0.972 | 0.051 | RandomUnderSampler(0.1) |
| `XGBoost_Undersample_EarlyStopping` ⭐ | **0.927** | 0.993 | 0.066 | საუკეთესო კომბინაცია |
| `XGBoost_EarlyStopping_complex` | 0.919 | 0.985 | 0.067 | lossguide policy, 2395 ხე |

---

#### ანალიზი run-ების მიხედვით

**Underfitted vs Overfitted:**

`XGBoost_Underfitted` (val: 0.867) — `max_depth=2` ნიშნავს რომ ხე მხოლოდ 2 დონის სიღრმეზე იყოფა. ასეთი მარტივი მოდელი ვერ ისწავლის fraud-ის რთულ პატერნებს. train AUC-იც (0.891) დაბალია — ეს კლასიკური **underfitting-ია**: მოდელი საკმარისად რთული არ არის.

`XGBoost_Overfitted` (val: 0.905, train: 1.000, gap: 0.095) — `max_depth=15` და regularization=0 იმდენად ამახსოვრებს სატრენინგო მონაცემებს რომ train AUC=1.0. val AUC კი baseline-ზე ნაკლებია. ეს კლასიკური **overfitting-ია**.

**Feature Engineering-ის გავლენა:**

`XGBoost_NoFE` (val: 0.913) vs `XGBoost_Baseline` (val: 0.919) — **+0.006 AUC** Feature Engineering-ისგან. ეს ნიშნავს რომ uid fingerprints, D ნორმალიზაცია, frequency encoding და aggregation ფიჩერები რეალურ სიგნალს შეიცავს.

**Early Stopping:**

`XGBoost_EarlyStopping` — learning_rate=0.01 + n_estimators=5000 + early_stopping_rounds=100. ოპტიმალური ხეების რაოდენობა = 3772. val AUC: 0.922. val AUC ყოველ 100 ხეზე იზრდებოდა ბოლომდე.

**Undersampling-ის გავლენა:**

`XGBoost_Undersampling` — RandomUnderSampler(sampling_strategy=0.1): fraud:non-fraud = 1:10. gap შემცირდა (0.064 → 0.051) — overfitting-ი შემცირდა, შედეგი გაუმჯობესდა.

**საუკეთესო კომბინაცია:**

`XGBoost_Undersample_EarlyStopping` (val: **0.927**) — undersampling + early stopping ერთდროულად:
- Undersampling: მონაცემების დასაბალანსებლად
- Early stopping: 3857 ოპტიმალური ხე (12000-დან) → მოდელი ოპტიმალურ მოდელს პოულობს
---

#### Feature Engineering-ის გავლენის შედარება

```
Feature Engineering გარეშე:  val AUC = 0.913  (125 ფიჩერი)
Feature Engineering-ით:      val AUC = 0.927  (181 ფიჩერი)
განსხვავება:                  +0.014 AUC
```
---

### ყველა მოდელის საბოლოო შედარება

| მოდელი | val AUC | მიდგომა |
|---|---|---|
| **XGBoost** ⭐ | **0.9271** | boosting + undersampling + early stopping |
| Random Forest | 0.9124 | bagging + class_weight='balanced' |
| Decision Tree | 0.8345 | single tree + undersampling |
| Logistic Regression | 0.8114 | linear model + L2 + scaling |

შედეგები ადასტურებს ML-ის ფუნდამენტურ პრინციპს: **model complexity უნდა შეესაბამებოდეს პრობლემის სტრუქტურას**. Fraud detection-ი non-linear, high-dimensional, time-series პრობლემაა — sequential boosting (XGBoost) ამ ამოცანისთვის ყველაზე შესაფერისი არქიტექტურაა.

--------------------------------------------------------------------------------


## საბოლოო მოდელის შერჩევა

**`XGBoost_Undersample_EarlyStopping`** არჩეულია საუკეთესო მოდელად:

| პარამეტრი | მნიშვნელობა |
|---|---|
| val AUC | **0.9271** |
| train AUC | 0.9932 |
| Gap | 0.0661 |
| n_estimators (ოპტიმალური) | 3857 |
| max_depth | 8 |
| learning_rate | 0.01 |
| grow_policy | lossguide |
| max_leaves | 64 |
| subsample / colsample_bytree | 0.85 / 0.85 |
| reg_alpha / reg_lambda | 0.1 / 2.0 |
| balancing | RandomUnderSampling(0.1) |

**val AUC 0.9271** — ყველა experiment-ს შორის საუკეთესო

---

## MLflow Tracking

**DagsHub MLflow:** [https://dagshub.com/gvakh23/ML_assignment2.mlflow](https://dagshub.com/gvakh23/ML_assignment2.mlflow)
**Kaggle Score: 0.90**
<img width="443" height="202" alt="image" src="https://github.com/user-attachments/assets/86882e6e-4c05-4243-8a2b-d72cb90a3984" />


### ექსპერიმენტების სტრუქტურა

```
XGBoost_Training
    ├── XGBoost_Cleaning               ← null threshold, fill value, M cols
    ├── XGBoost_Categorical_Encoding   ← OHE cols, LE cols, encoders
    ├── XGBoost_Feature_Engineering    ← ახალი ფიჩერები, freq maps, agg maps
    ├── XGBoost_Feature_Selection      ← variance threshold, V group selection
    ├── XGBoost_Underfitted            ← underfitting demonstration
    ├── XGBoost_Overfitted             ← overfitting demonstration
    ├── XGBoost_NoFE                   ← FE-ის გავლენის proof
    ├── XGBoost_Baseline               ← starting point
    ├── XGBoost_Tuned                  ← hyperparameter tuning
    ├── XGBoost_EarlyStopping          ← optimal tree count
    ├── XGBoost_Undersampling          ← imbalance handling
    ├── XGBoost_Undersample_EarlyStopping ← საუკეთესო
    └── XGBoost_FullPipeline_Registry  ← full pipeline, registered
```

### Model Registry

საუკეთესო მოდელი დარეგისტრირებულია **`BestFraudDetector`** სახელით Model Registry-ში.

Pipeline-ი პირდაპირ **raw, დაუმუშავებელ** მონაცემებზე მუშაობს:
```
raw test data → Cleaner → CategoricalEncoder → FeatureEngineer → TargetDropper → FeatureSelector → XGBClassifier → fraud probability
```
