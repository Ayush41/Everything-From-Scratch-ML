import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "playground-series-s6e5")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "submission.csv")
SEED = 42


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def create_features(df):
    df = df.copy()
    df["Position_Change_abs"] = df["Position_Change"].abs()
    df["LapTime_per_Position"] = df["LapTime (s)"] / df["Position"].replace(0, np.nan)
    df["LapTime_per_Position"] = df["LapTime_per_Position"].fillna(df["LapTime (s)"])
    df["TyreLife_ratio"] = df["TyreLife"] / (df["LapNumber"] + 1e-6)
    df["RaceProgress_sq"] = df["RaceProgress"] ** 2
    df["LapTime_Delta_abs"] = df["LapTime_Delta"].abs()
    df["RaceProgress_bin"] = pd.cut(df["RaceProgress"], bins=[-1, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False).astype(np.int8)
    df["Position_bin"] = pd.cut(df["Position"], bins=[0, 5, 10, 15, 20, 25, 30, 50], labels=False).astype(np.int8)
    df["LapNumber_bin"] = pd.cut(df["LapNumber"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 100], labels=False).astype(np.int8)
    df["Position_Change_sign"] = (df["Position_Change"] > 0).astype(np.int8)
    df["LapTime_Delta_sign"] = (df["LapTime_Delta"] > 0).astype(np.int8)
    df["Compound_short"] = df["Compound"].astype(str).str.upper().fillna("UNK")
    return df


def add_target_encoding(train, test):
    train = train.copy()
    test = test.copy()
    train["RaceYear"] = train["Race"].astype(str) + "_" + train["Year"].astype(str)
    test["RaceYear"] = test["Race"].astype(str) + "_" + test["Year"].astype(str)

    groups = [
        ["Driver"],
        ["Compound"],
        ["RaceYear"],
        ["Race"],
        ["Stint"],
        ["Position_bin"],
        ["LapNumber_bin"],
    ]

    for group in groups:
        name = "_".join(group)
        stats = train.groupby(group)["PitNextLap"].agg(["mean", "count"]).reset_index()
        stats.columns = group + [f"{name}_pit_rate", f"{name}_count"]
        train = train.merge(stats, on=group, how="left")
        test = test.merge(stats, on=group, how="left")
        test[f"{name}_pit_rate"] = test[f"{name}_pit_rate"].fillna(train["PitNextLap"].mean())
        test[f"{name}_count"] = test[f"{name}_count"].fillna(0)

    train["Compound_short"] = train["Compound_short"].fillna("UNK")
    test["Compound_short"] = test["Compound_short"].fillna("UNK")
    return train, test


def encode_categorical(train, test, categorical_columns):
    for col in categorical_columns:
        train[col] = train[col].astype(str).fillna("nan")
        test[col] = test[col].astype(str).fillna("nan")
        labels, _ = pd.factorize(pd.concat([train[col], test[col]], axis=0), sort=True)
        train[col] = labels[: len(train)]
        test[col] = labels[len(train) :]
    return train, test


def build_feature_set(train, test):
    train = create_features(train)
    test = create_features(test)
    train, test = add_target_encoding(train, test)

    categorical_columns = [
        "Driver",
        "Compound",
        "Race",
        "RaceYear",
        "Compound_short",
        "Position_bin",
        "LapNumber_bin",
    ]
    train, test = encode_categorical(train, test, categorical_columns)

    features = [
        "Driver",
        "Compound",
        "Race",
        "Year",
        "PitStop",
        "LapNumber",
        "Stint",
        "TyreLife",
        "Position",
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "RaceProgress",
        "Position_Change",
        "Position_Change_abs",
        "LapTime_per_Position",
        "TyreLife_ratio",
        "RaceProgress_sq",
        "LapTime_Delta_abs",
        "RaceProgress_bin",
        "Position_bin",
        "LapNumber_bin",
        "Position_Change_sign",
        "LapTime_Delta_sign",
        "Compound_short",
    ]

    for group in ["Driver", "Compound", "RaceYear", "Race", "Stint", "Position_bin", "LapNumber_bin"]:
        features += [f"{group}_pit_rate", f"{group}_count"]

    features = [f for f in features if f in train.columns]
    return train, test, features


def train_and_predict(train, test, features):
    y = train["PitNextLap"].values
    X = train[features]
    X_test = test[features]
    groups = train["RaceYear"].values

    oof_preds_xgb = np.zeros(len(train), dtype=np.float32)
    oof_preds_lgb = np.zeros(len(train), dtype=np.float32)
    test_preds_xgb = np.zeros(len(test), dtype=np.float32)
    test_preds_lgb = np.zeros(len(test), dtype=np.float32)

    cv = GroupKFold(n_splits=5)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {fold + 1}/5")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params_xgb = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "seed": SEED,
            "nthread": 4,
            "tree_method": "hist",
        }
        xgb_model = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        oof_preds_xgb[val_idx] = xgb_model.predict(dval, ntree_limit=xgb_model.best_ntree_limit)
        test_preds_xgb += xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit) / cv.n_splits

        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            n_estimators=1000,
            num_leaves=31,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=4,
        )
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / cv.n_splits

        fold_score = roc_auc_score(y_val, 0.5 * oof_preds_xgb[val_idx] + 0.5 * oof_preds_lgb[val_idx])
        fold_scores.append(fold_score)
        print(f"  XGB valid AUC: {roc_auc_score(y_val, oof_preds_xgb[val_idx]):.5f}")
        print(f"  LGB valid AUC: {roc_auc_score(y_val, oof_preds_lgb[val_idx]):.5f}")
        print(f"  Blend valid AUC: {fold_score:.5f}\n")

    blended_test_preds = 0.5 * test_preds_xgb + 0.5 * test_preds_lgb
    oof_blend = 0.5 * oof_preds_xgb + 0.5 * oof_preds_lgb
    print("CV blend AUC:", roc_auc_score(y, oof_blend))
    print("Fold scores:", [round(s, 5) for s in fold_scores])
    return blended_test_preds


def save_submission(test, preds, output_path):
    submission = pd.DataFrame({"id": test["id"], "PitNextLap": preds})
    submission["PitNextLap"] = submission["PitNextLap"].clip(0, 1)
    submission.to_csv(output_path, index=False)
    print("Saved submission to", output_path)


def main():
    train, test = load_data()
    train, test, features = build_feature_set(train, test)
    print("Using features:", features)
    preds = train_and_predict(train, test, features)
    save_submission(test, preds, OUTPUT_PATH)


if __name__ == "__main__":
    main()
'''
with open('train_submit.py', 'w') as f:
    f.write(content)
print('wrote', len(content), 'chars')
PY
'''
