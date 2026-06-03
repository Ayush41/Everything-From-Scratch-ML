import joblib
import pandas as pd
import os

_model_cache = None


def default_model_path():
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, 'models', 'xgb_pit_joblib.pkl')


def load_model(path=None):
    if path is None:
        path = default_model_path()
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _model_cache = joblib.load(path)
    return _model_cache


def preprocess_input(df):
    # mirror features used in training
    df = df.copy()
    if 'laps_on_tyre' not in df.columns:
        df['laps_on_tyre'] = df.get('laps_on_tyre', 1)
    if 'lap_number' in df.columns and 'race_id' in df.columns:
        df['lap_norm'] = df['lap_number'] / df.groupby('race_id')['lap_number'].transform('max')
    else:
        df['lap_norm'] = df.get('lap_norm', 0.5)
    if 'avg_last3' not in df.columns and 'lap_time' in df.columns:
        df['avg_last3'] = df['lap_time']
    if 'tyre_compound' in df.columns and 'is_fast_tyre' not in df.columns:
        df['is_fast_tyre'] = df['tyre_compound'].map({'soft':2,'medium':1,'hard':0}).fillna(1)
    for c in ['gap_to_leader', 'lap_time']:
        if c not in df.columns:
            df[c] = 0.0
    return df


def predict_dataframe(df, model_path=None):
    m = load_model(model_path)
    model = m['model']
    features = m['features']
    df2 = preprocess_input(df)
    X = df2[features]
    probs = model.predict_proba(X)[:, 1]
    return probs


def predict_single(lap_number=20, lap_time=85.3, tyre_compound='medium', gap_to_leader=10.0, laps_on_tyre=5, model_path=None):
    df = pd.DataFrame([{
        'lap_number': lap_number,
        'lap_time': lap_time,
        'tyre_compound': tyre_compound,
        'gap_to_leader': gap_to_leader,
        'laps_on_tyre': laps_on_tyre
    }])
    probs = predict_dataframe(df, model_path)
    m = load_model(model_path)
    th = m.get('threshold', 0.5)
    prob = float(probs[0])
    return {'pit_probability': prob, 'will_pit_next': bool(prob > th), 'threshold': th}
