# In your training runs, compare:
# Run 1: scale_pos_weight=28 (recommended)
# Run 2: scale_pos_weight=1 + SMOTE (show it's worse)
# This demonstrates you understand the tradeoff

My honest opinion though
For tree models specifically, -999 is almost always better than mean/median. Here's why:

Mean/median imputation tells the model "this missing transaction looked average" — which is wrong and destroys the missingness signal
-999 tells the model "this value was missing" — the tree can learn a separate branch for it
In this dataset specifically, whether a V column is missing or not is itself a strong fraud signal — a transaction that didn't trigger Vesta's module is different from one that did

XGBoost_Training experiment
    ├── XGBoost_Cleaning
    ├── XGBoost_Categorical_Encoding
    ├── XGBoost_Feature_Engineering
    ├── XGBoost_Feature_Selection
    ├── XGBoost_Underfitted          ✅ done — val: 0.867
    ├── XGBoost_Overfitted           ✅ done — val: 0.905
    ├── XGBoost_Baseline             ✅ done — val: 0.919
    ├── XGBoost_NoFE                 ← run next, proves FE value
    ├── XGBoost_Tuned                ← reduce gap below 0.03
    ├── XGBoost_NoClassWeight        ← prove scale_pos_weight matters
    └── XGBoost_BestModel_Pipeline   ← save best for registry



    The tuning improved within-period generalization (CV went up) but the val set represents a different distribution — 
    slightly different fraud patterns from a later time period — so the improvement didn't transfer.

    XGBoost_Training experiment
    ├── XGBoost_Cleaning
    ├── XGBoost_Categorical_Encoding  
    ├── XGBoost_Feature_Engineering
    ├── XGBoost_Feature_Selection
    ├── XGBoost_Underfitted          val: 0.867
    ├── XGBoost_Overfitted           val: 0.905
    ├── XGBoost_NoFE                 val: 0.913  (proves FE value)
    ├── XGBoost_Baseline             val: 0.919
    ├── XGBoost_Tuned                val: 0.919  (tuning hit ceiling)
    ├── XGBoost_EarlyStopping_3000   val: 0.922
    ├── XGBoost_EarlyStopping_5000   val: 0.924  (best)
    └── XGBoost_BestModel_Pipeline   ← save this
