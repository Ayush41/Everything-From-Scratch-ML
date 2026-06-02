# Predicting F1 Pit Stops
Playground Series - Season 6 Episode 5

Welcome to this project repository for the Kaggle Playground Series task: predict whether a Formula 1 driver will pit on the next lap. This `Info.md` documents the problem, the data, the approach I used, and the steps to reproduce and deploy a small Gradio app (Hugging Face Spaces).

## Overview
- Goal: Predict whether a driver will make a pit stop on the next lap (binary classification).
- Scope: lap-level telemetry and race-context features; avoid temporal leakage and respect group structure (races/drivers).
- Primary metrics: ROC AUC, PR AUC, and F1 at a selected threshold. Calibration and precision/recall balance are important because pit events are rare.

## Problem Statement
Given the available features for the current lap (driver, team, lap number, tyre compound, tyre age, lap time, gaps, race phase, etc.), produce a probability that the driver will pit on the following lap. The model should be trained only on information that would be available at prediction time (no future lap info).

Key constraints and considerations:
- Avoid future-data leakage: do not include features computed using future laps.
- Grouped validation: use group-aware splits (by `race_id` or `race_id + driver_id`) to simulate deployment.
- Class imbalance: pit events are sparse — use class weighting, up/down-sampling, or `scale_pos_weight` in tree models.

## Dataset
Dataset files (examples in this workspace):
- `playground-series-s6e5/train.csv` — training laps with `pit_next` label.
- `playground-series-s6e5/test.csv` — test laps for submission.
- `playground-series-s6e5/sample_submission.csv` — submission format.

Inspect data first to identify available columns and missing values, then build features.

## Exploratory Data Analysis & Feature Engineering (Summary)
- Basic checks: distributions of lap counts, pit frequency per race/driver, missingness.
- Temporal features: `lap_number`, `lap_norm` (lap / max_laps), `stint_progress`.
- Tyre features: `tyre_compound` (ordinal mapping), `laps_on_tyre` (cumcount), `tyre_age`.
- Rolling aggregates: `avg_last3_lap_time`, `std_last5_lap_time`.
- Race context: `gap_to_leader`, `position`, `is_safety_car_lap` if available, `weather` proxies.
- Categorical handling: one-hot or target encoding for `driver_id` / `team` (careful with leakage).

Example feature creation (concept):
```python
import pandas as pd

def add_features(df):
	df = df.copy()
	df['laps_on_tyre'] = df.groupby(['race_id','driver_id','tyre_set_id']).cumcount() + 1
	df['lap_norm'] = df['lap_number'] / df.groupby('race_id')['lap_number'].transform('max')
	df['avg_last3'] = (
		df.groupby(['race_id','driver_id'])['lap_time']
		  .rolling(3, min_periods=1)
		  .mean()
		  .reset_index(level=[0,1], drop=True)
	)
	df['is_fast_tyre'] = df['tyre_compound'].map({'soft':2, 'medium':1, 'hard':0})
	return df

# usage
train = pd.read_csv('playground-series-s6e5/train.csv')
train = add_features(train)
```

## Modeling Approach
- Baseline: Logistic Regression with class weights and basic features.
- Main approach: Gradient boosted trees (XGBoost or LightGBM) with careful validation and early stopping.
- Calibration: apply Platt scaling or isotonic calibration if probabilities are miscalibrated.
- Validation: `GroupKFold` by `race_id` (or `race_id + driver_id`) to simulate unseen races.

Example grouped split:
```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
groups = train['race_id']
for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train['pit_next'], groups)):
	tr = train.iloc[tr_idx]
	val = train.iloc[val_idx]
	# train model on tr, evaluate on val
```

## Training (example with XGBoost)
```python
import xgboost as xgb
features = ['lap_norm','laps_on_tyre','avg_last3','gap_to_leader','is_fast_tyre']
dtrain = xgb.DMatrix(train[features], label=train['pit_next'])
params = {
	'objective':'binary:logistic',
	'eval_metric':'auc',
	'eta':0.05,
	'max_depth':6,
	'scale_pos_weight': 5  # tune according to imbalance
}
bst = xgb.train(params, dtrain, num_boost_round=2000, early_stopping_rounds=50, evals=[(dtrain,'train')])
bst.save_model('models/xgb_pit.model')
```

## Evaluation
Use ROC AUC and PR AUC as primary metrics. Tune a probability threshold on validation to maximize F1 for the final binary decision.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
preds = bst.predict(xgb.DMatrix(val[features]))
print('ROC AUC', roc_auc_score(val['pit_next'], preds))
print('PR AUC', average_precision_score(val['pit_next'], preds))

# find best threshold for F1 (simple search)
best_f1 = 0
best_th = 0.5
for th in [i/100 for i in range(1,100)]:
	f1 = f1_score(val['pit_next'], preds > th)
	if f1 > best_f1:
		best_f1, best_th = f1, th
print('Best F1', best_f1, 'threshold', best_th)
```

## Inference & Submission
```python
test = pd.read_csv('playground-series-s6e5/test.csv')
test = add_features(test)
test_preds = bst.predict(xgb.DMatrix(test[features]))
sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')
sub['pit_next'] = test_preds
sub.to_csv('submission.csv', index=False)
```

## Gradio App (for Hugging Face Spaces)
Below is a lightweight `app.py` to expose the model via Gradio. Place `app.py` and the trained `models/xgb_pit.model` in the Space repo.

```python
import gradio as gr
import xgboost as xgb
import pandas as pd

model = xgb.Booster()
model.load_model('models/xgb_pit.model')

def predict_pit(lap_number, lap_time, tyre_compound, gap_to_leader, laps_on_tyre):
	df = pd.DataFrame([{
		'lap_number': lap_number,
		'lap_time': lap_time,
		'tyre_compound': tyre_compound,
		'gap_to_leader': gap_to_leader,
		'laps_on_tyre': laps_on_tyre
	}])
	df['lap_norm'] = df['lap_number'] / 58
	df['is_fast_tyre'] = {'soft':2,'medium':1,'hard':0}.get(tyre_compound,1)
	features = ['lap_norm','laps_on_tyre','lap_time','gap_to_leader','is_fast_tyre']
	d = xgb.DMatrix(df[features])
	prob = model.predict(d)[0]
	return {"pit_probability": float(prob), "will_pit_next": bool(prob > 0.5)}

iface = gr.Interface(
	fn=predict_pit,
	inputs=[
		gr.Number(label='Lap number', value=20),
		gr.Number(label='Lap time (s)', value=85.3),
		gr.Dropdown(['soft','medium','hard'], label='Tyre compound', value='medium'),
		gr.Number(label='Gap to leader (s)', value=12.4),
		gr.Number(label='Laps on tyre', value=6)
	],
	outputs=[gr.Label(num_top_classes=2)],
	title='F1 Pit Stop Predictor'
)

if __name__ == '__main__':
	iface.launch(server_name='0.0.0.0', server_port=7860)
```

## requirements.txt (for the Space)
```
pandas
scikit-learn
xgboost
gradio
joblib
```

## What I did & How I did it
- Loaded lap-level data, merged any driver/team metadata, and inspected distributions.
- Engineered lap- and stint-level features that track tyre age and recent performance.
- Used grouped cross-validation to avoid race-level leakage.
- Trained a gradient-boosted classifier (XGBoost) with class-balance adjustments and early stopping.
- Packaged inference and a small Gradio UI for easy experimentation and deployment to Hugging Face Spaces.

## Deployment
1. Create a new Hugging Face Space under your account (e.g., `huggingface.co/spaces/<your-username>/f1-pitstop`).
2. Add the following to the Space repo: `app.py`, `models/xgb_pit.model`, `requirements.txt` and this `README`/`Info.md`.
3. Push to the Space; it will build and host the Gradio app.

Deployment link (replace with your username):

https://huggingface.co/spaces/<your-username>/f1-pitstop

## Files referenced in this workspace
- `playground-series-s6e5/train.csv`, `playground-series-s6e5/test.csv`, `playground-series-s6e5/sample_submission.csv`
- `f1-PitStop/train_submit.py`, `f1-PitStop/train_submit_final.py`, `f1-PitStop/f1-PitStop.py`

---
If you want, I can now add the `app.py` and `requirements.txt` files to the repo and place a small example `models/xgb_pit.model` placeholder. Would you like me to create those now?

