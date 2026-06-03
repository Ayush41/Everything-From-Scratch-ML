import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier


def normalize_columns(df):
    df = df.copy()
    # standardize dataset column names to training-friendly names
    column_map = {
        'PitNextLap': 'pit_next',
        'PitStop': 'pit_stop',
        'LapNumber': 'lap_number',
        'LapTime (s)': 'lap_time',
        'TyreLife': 'tyre_age',
        'Compound': 'tyre_compound',
        'Race': 'race_id',
        'Driver': 'driver_id',
        'RaceProgress': 'race_progress',
        'Position': 'position',
        'LapTime_Delta': 'lap_time_delta',
        'Cumulative_Degradation': 'cum_degradation',
        'Position_Change': 'position_change'
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    return df


def add_features(df):
    df = normalize_columns(df)
    if 'race_id' in df.columns and 'driver_id' in df.columns and 'stint' in df.columns:
        df['laps_on_tyre'] = df.groupby(['race_id', 'driver_id', 'stint']).cumcount() + 1
    elif 'race_id' in df.columns and 'driver_id' in df.columns:
        df['laps_on_tyre'] = df.groupby(['race_id', 'driver_id']).cumcount() + 1
    else:
        df['laps_on_tyre'] = df.get('laps_on_tyre', 1)

    if 'race_id' in df.columns and 'lap_number' in df.columns:
        df['lap_norm'] = df['lap_number'] / df.groupby('race_id')['lap_number'].transform('max')
    else:
        df['lap_norm'] = df.get('lap_norm', 0.5)

    if 'lap_time' in df.columns and 'race_id' in df.columns and 'driver_id' in df.columns:
        df['avg_last3'] = (
            df.groupby(['race_id', 'driver_id'])['lap_time']
              .rolling(3, min_periods=1)
              .mean()
              .reset_index(level=[0,1], drop=True)
        )
    else:
        df['avg_last3'] = df.get('lap_time', 0.0)

    if 'tyre_compound' in df.columns:
        df['tyre_compound'] = df['tyre_compound'].astype(str).str.lower()
        df['is_fast_tyre'] = df['tyre_compound'].map({'soft': 2, 'medium': 1, 'hard': 0}).fillna(1)
    else:
        df['is_fast_tyre'] = 1

    for c in ['gap_to_leader', 'lap_time', 'race_progress', 'position', 'lap_time_delta']:
        if c not in df.columns:
            df[c] = 0.0

    return df


def main():
    base_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    train_path = os.path.join(base_dir, 'playground-series-s6e5', 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)

    train = pd.read_csv(train_path)
    train = add_features(train)

    features = ['lap_norm', 'laps_on_tyre', 'avg_last3', 'gap_to_leader', 'is_fast_tyre', 'lap_time']
    features = [f for f in features if f in train.columns]

    X = train[features]
    y = train['pit_next'] if 'pit_next' in train.columns else train.get('target', None)
    if y is None:
        raise ValueError('No label column found (pit_next)')

    groups = train['race_id'] if 'race_id' in train.columns else None
    if groups is None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        gkf = GroupKFold(n_splits=5)
        tr_idx, val_idx = next(iter(gkf.split(X, y, groups)))
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = max(1.0, neg / max(1.0, pos))

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train, verbose=False)
    val_preds = model.predict_proba(X_val)[:, 1]
    print('ROC AUC:', roc_auc_score(y_val, val_preds))
    print('PR AUC:', average_precision_score(y_val, val_preds))

    best_f1 = 0.0
    best_th = 0.5
    for th in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_val, val_preds > th)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    print(f'Best F1 on validation: {best_f1:.4f} at threshold {best_th:.2f}')

    model.fit(X, y)

    out_path = os.path.join(base_dir, 'models', 'xgb_pit_joblib.pkl')
    joblib.dump({'model': model, 'features': features, 'threshold': float(best_th)}, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    main()
