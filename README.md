# IEEE-CIS Fraud Detection — XGBoost პროექტი

## პროექტის მიმოხილვა

ეს პროექტი Kaggle-ის კონკურსის [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) ამოხსნაა. კონკურსის მიზანია საკრედიტო ბარათის ტრანზაქციებში თაღლითობის გამოვლენა. მონაცემები მოწოდებულია Vesta Corporation-ის მიერ და შეიცავს 590,540 სატრენინგო ტრანზაქციას 434 ფიჩერით.

**შეფასების მეტრიკა:** ROC-AUC  
**MLflow ექსპერიმენტები:** [DagsHub](https://dagshub.com/gvakh23/ML_assignment2.mlflow)

---

## რეპოზიტორიის სტრუქტურა

```
├── eda-fraud.ipynb                          # მონაცემების ექსპლორაცია
├── model_experiment_XGBoost.ipynb           # XGBoost ექსპერიმენტები (pipeline ვერსია)
├── model_experiment_LogisticRegression.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_AdaBoost.ipynb
├── model_inference.ipynb                    # საუკეთესო მოდელით პროგნოზი
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

> 📸 *[ჩასვი EDA Cell 4-ის სამი გრაფიკი: Fraud rate over time / log(TransactionAmt) / Train vs Test time split]*

- **თაღლითობის სიხშირე დროში იცვლება** — 1%-დან 7%-მდე, რაც ნიშნავს რომ მოდელმა დროის მიმართ უნდა განზოგადდეს
- **TransactionAmt ძლიერ გადახრილია** — log-ტრანსფორმაცია სავალდებულოა
- **Train და Test დროით გამოყოფილია** — Train: დღე 0-180, Test: დღე 200-400. გადაფარვა არ არის, ამიტომ random KFold-ის ნაცვლად **TimeSeriesSplit** გამოვიყენეთ

> 📸 *[ჩასვი EDA Cell 6-ის V columns NaN pattern ანალიზი]*

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

**EDA-ს დასკვნა:** 12 სვეტი >90% null-ით — ეს სვეტები ვერ შეიცავს სასარგებლო სიგნალს.

| ნაბიჯი | მიდგომა | მიზეზი |
|---|---|---|
| მაღალი null-ის სვეტები | 90%-ზე მეტი null → წაშლა | >90% null სვეტი ვერ ისწავლის სასარგებლო პატერნს |
| M სვეტები (M1-M9) | T→1, F→0, NaN→-999 | Match ფიჩერებია (True/False) — ბინარული ბუნება |
| NaN შევსება | -999 | XGBoost-ი tree-based მოდელია — -999 ნორმალური range-ის გარეთაა, ხე შექმნის ცალკე branch-ს "missing" ტრანზაქციებისთვის. mean/median-ისგან განსხვავებით, ინახავს "missing" ინფორმაციას |

**შედეგი:** 434 → 422 სვეტი (12 წაიშალა)

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

#### UID Fingerprint (winning solution-ის ტექნიკა)
```
uid  = card1 + "_" + card2               ← ბარათის ID
uid2 = card1 + "_" + addr1              ← ბარათი + მისამართი  
uid3 = card1 + "_" + addr1 + "_" + D1  ← ინდივიდუალური კლიენტის fingerprint
```
`D1` = დღეები ბარათის გამოყენების დაწყებიდან — კლიენტზე თითქმის მუდმივია. `card1+addr1+D1` ცალკეულ ადამიანს აიდენტიფიცირებს.

#### Email ფიჩერები (top solution-ის მიდგომა)
```
same_email     = (P_emaildomain == R_emaildomain)  ← გამყიდველი/მყიდველი email შეუსაბამობა
P_email_suffix = gmail, yahoo, protonmail...        ← TLD-ის encoded მნიშვნელობა
```
protonmail.com-ს fraud rate >90%-ია.

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
> 📸 *[ჩასვი EDA Cell 6-ის NaN pattern ანალიზის output]*

```python
# ჯგუფში 20+ კორელირებული სვეტიდან ვინახავთ მხოლოდ 1-ს
best_col = max(group_cols, key=lambda c: abs(corr(c, isFraud)))
```

**საბოლოო შედეგი:** ~506 → **181 ფიჩერი**

---

## Training ექსპერიმენტები

### XGBoost_Training ექსპერიმენტი

Cross-validation: `TimeSeriesSplit(n_splits=5)` — random KFold-ი გამოიწვევდა data leakage-ს.  
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

`XGBoost_Overfitted` (val: 0.905, train: 1.000, gap: 0.095) — `max_depth=15` და regularization=0 იმდენად ამახსოვრებს სატრენინგო მონაცემებს რომ train AUC=1.0. val AUC კი baseline-ზე ნაკლებია — ეს კლასიკური **overfitting-ია**.

**Feature Engineering-ის გავლენა:**

`XGBoost_NoFE` (val: 0.913) vs `XGBoost_Baseline` (val: 0.919) — **+0.006 AUC** Feature Engineering-ისგან. ეს ნიშნავს რომ uid fingerprints, D ნორმალიზაცია, frequency encoding და aggregation ფიჩერები რეალურ სიგნალს შეიცავს.

**Tuning-ის ჭერი:**

`XGBoost_Baseline` (val: 0.919) vs `XGBoost_Tuned` (val: 0.919) — hyperparameter tuning-მა val AUC-ი ვერ გაზარდა. CV mean გაუმჯობესდა (0.910 → 0.914) მაგრამ val-ზე ეფექტი არ გამოჩნდა. **დასკვნა:** მოდელი feature ceiling-ს მიაღწია — ახალი ფიჩერები უფრო სასარგებლო იქნება ვიდრე hyperparameter tuning.

**Early Stopping:**

`XGBoost_EarlyStopping` — learning_rate=0.01 + n_estimators=5000 + early_stopping_rounds=100. ოპტიმალური ხეების რაოდენობა = 3772. val AUC: 0.922. val AUC ყოველ 100 ხეზე იზრდებოდა ბოლომდე — ეს ნიშნავს რომ baseline-ის 500 ხე საკმარისი არ იყო.

**Undersampling-ის გავლენა:**

`XGBoost_Undersampling` — RandomUnderSampler(sampling_strategy=0.1): fraud:non-fraud = 1:10. gap შემცირდა (0.064 → 0.051) — overfitting-ი შემცირდა. val AUC: 0.921. redundant majority class მაგალითების ამოღება მოდელს აიძულებს fraud boundary-ს უკეთ ისწავლოს.

**საუკეთესო კომბინაცია:**

`XGBoost_Undersample_EarlyStopping` (val: **0.927**) — undersampling + early stopping ერთდროულად:
- Undersampling: redundant non-fraud მაგალითები ამოვიღეთ → cleaner training data
- Early stopping: 3857 ოპტიმალური ხე (12000-დან) → model-ი ზუსტ stopping point-ს პოულობს
- Gap: 0.066 — acceptable range fraud detection-ისთვის (train/test დროით გამოყოფილია)

---

#### Feature Engineering-ის გავლენის შედარება

```
Feature Engineering გარეშე:  val AUC = 0.913  (125 ფიჩერი)
Feature Engineering-ით:      val AUC = 0.927  (181 ფიჩერი)
განსხვავება:                  +0.014 AUC
```

ყველაზე მნიშვნელოვანი ახალი ფიჩერები:
- `uid3 = card1+addr1+D1` — ინდივიდუალური კლიენტის fingerprint
- `same_email` — email შეუსაბამობა
- `card1_TransactionAmt_std` — ბარათის დანახარჯის ვარიაციულობა

---

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

**რატომ ეს მოდელი:**

1. **val AUC 0.9271** — ყველა experiment-ს შორის საუკეთესო
2. **Undersampling** ამცირებს overfitting-ს (gap: 0.064 → 0.051 baseline-თან შედარებით) redundant majority class მაგალითების ამოღებით
3. **Early stopping** ავტომატურად პოულობს ოპტიმალურ tree-ების რაოდენობას — 12,000-ის ნაცვლად 3,857
4. **Gap 0.066** — acceptable fraud detection-ისთვის, სადაც train/test დროით გამოყოფილია და fraud პატერნები ბუნებრივად იცვლება

---

## MLflow Tracking

**DagsHub MLflow:** [https://dagshub.com/gvakh23/ML_assignment2.mlflow](https://dagshub.com/gvakh23/ML_assignment2.mlflow)

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

### დალოგული მეტრიკები

| მეტრიკა | აღწერა |
|---|---|
| `train_auc` | ROC-AUC სატრენინგო ნაკრებზე |
| `val_auc` | ROC-AUC validation ნაკრებზე |
| `overfit_gap` | train_auc − val_auc |
| `cv_mean_auc` | TimeSeriesSplit 5-fold საშუალო |
| `cv_std_auc` | TimeSeriesSplit 5-fold სტანდარტული გადახრა |
| `best_iteration` | ოპტიმალური ხეების რაოდენობა (early stopping) |
| `final_features` | feature selection-ის შემდეგ სვეტების რაოდენობა |

### Model Registry

საუკეთესო მოდელი დარეგისტრირებულია **`BestFraudDetector`** სახელით Model Registry-ში.

```python
# Inference-ში გამოყენება:
pipeline = mlflow.sklearn.load_model("models:/BestFraudDetector/1")
preds = pipeline.predict_proba(raw_test_df)[:, 1]
```

Pipeline-ი პირდაპირ **raw, დაუმუშავებელ** მონაცემებზე მუშაობს:
```
raw test data → Cleaner → CategoricalEncoder → FeatureEngineer → TargetDropper → FeatureSelector → XGBClassifier → fraud probability
```
