"""Stage 9: freeze a 20-case prospective final holdout before running EnergyPlus.
Usage: python stage9_prepare.py
Inputs: dataset.csv, adaptive_results.csv, validation_results.csv,
        stage8_results/frozen_selected_classifier.joblib,
        stage7_results/frozen_HistGradientBoosting.joblib
Outputs: stage9_plan/final_manifest.csv and frozen model copies.
"""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'stage9_plan'; OUT.mkdir(exist_ok=True)
FEATURES=['Heatwave_Intensity_C','Heatwave_Duration_day','IT_Load_Wm2','Capacity_Ratio','Cooling_Setpoint_C']
SEED=94023
rng=np.random.default_rng(SEED)

def load(name):
    p=ROOT/name
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_csv(p,encoding='utf-8-sig')

old=pd.concat([load('dataset.csv')[FEATURES],load('adaptive_results.csv')[FEATURES],load('validation_results.csv')[FEATURES]],ignore_index=True).astype(float)
clf_path=ROOT/'stage8_results/frozen_selected_classifier.joblib'
reg_path=ROOT/'stage7_results/frozen_HistGradientBoosting.joblib'
if not clf_path.exists() or not reg_path.exists():
    raise FileNotFoundError('Run Stage 8 and Stage 7 locally first; frozen model files are missing.')
clf_obj=joblib.load(clf_path)
reg_obj=joblib.load(reg_path)
print('Classifier model metadata:',list(clf_obj) if isinstance(clf_obj,dict) else type(clf_obj).__name__)
print('Regression model metadata:',list(reg_obj) if isinstance(reg_obj,dict) else type(reg_obj).__name__)
clf=clf_obj['model'] if isinstance(clf_obj,dict) else clf_obj
reg=reg_obj['regressor'] if isinstance(reg_obj,dict) and 'regressor' in reg_obj else None
if reg is None and isinstance(reg_obj,dict):
    print('Regression bundle keys:',list(reg_obj))
    for k in ['reg','tmax_regressor','regression_model']:
        if k in reg_obj: reg=reg_obj[k];break
if reg is None: raise ValueError('Cannot locate Tmax regressor in Stage 7 model bundle; inspect its keys.')

def sample(n):
    return pd.DataFrame({'Heatwave_Intensity_C':rng.uniform(0,10,n),
        'Heatwave_Duration_day':rng.integers(1,6,n),
        'IT_Load_Wm2':rng.uniform(300,1000,n),
        'Capacity_Ratio':rng.uniform(.9,1.3,n),
        'Cooling_Setpoint_C':rng.uniform(20,27,n)})

# Reject nearly identical points in normalized five-dimensional feature space.
scale=np.array([10,4,700,.4,7.])
existing=old[FEATURES].to_numpy(float)
def novel(row, selected):
    a=row[FEATURES].to_numpy(dtype=float)
    other=np.vstack([existing,selected]) if len(selected) else existing
    return np.min(np.linalg.norm((other-a)/scale,axis=1))>.025

# Generalization set: random coverage, selected without using target labels.
general=[]
for _,r in sample(2000).iterrows():
    if novel(r,np.asarray(general).reshape(-1,5) if general else []):
        general.append(r[FEATURES].to_numpy(float))
    if len(general)==10:break
if len(general)!=10:raise RuntimeError('Could not find 10 novel general cases')
# Boundary set: deliberately difficult but chosen solely using frozen model predictions.
candidates=sample(12000)
probs=clf.predict_proba(candidates[FEATURES]); labels=list(clf.classes_)
# Ambiguous multiclass predictions (low difference between largest two probabilities)
margin=np.sort(probs,axis=1)[:,-1]-np.sort(probs,axis=1)[:,-2]
chosen=[]
for i in np.argsort(margin):
    r=candidates.iloc[i]
    if novel(r,np.vstack([general,chosen]) if chosen else np.asarray(general)):
        chosen.append(r[FEATURES].to_numpy(float))
    if len(chosen)==10:break
if len(chosen)!=10:raise RuntimeError('Could not find 10 novel boundary cases')
mat=np.vstack([general,chosen]); plan=pd.DataFrame(mat,columns=FEATURES)
plan['Heatwave_Duration_day']=plan['Heatwave_Duration_day'].astype(int)
plan['Heatwave_Intensity_C']=plan['Heatwave_Intensity_C'].round(3)
plan['IT_Load_Wm2']=plan['IT_Load_Wm2'].round(3)
plan['Capacity_Ratio']=plan['Capacity_Ratio'].round(4)
plan['Cooling_Setpoint_C']=plan['Cooling_Setpoint_C'].round(3)
plan.insert(0,'Validation_ID',[f'FINAL_{i:03d}' for i in range(1,21)])
plan.insert(1,'Holdout_Group',['General']*10+['Boundary']*10)
plan['Predicted_Risk_Class']=clf.predict(plan[FEATURES])
plan['Predicted_Failure_Score_Uncalibrated']=clf.predict_proba(plan[FEATURES])[:,labels.index('Failure')]
plan['Predicted_Tmax_C_Operational_Model']=reg.predict(plan[FEATURES])
plan['Expected_Available_Capacity_kW']=(68.089*plan.IT_Load_Wm2/600*plan.Capacity_Ratio*(1-.02*plan.Heatwave_Intensity_C)).round(3)
plan.to_csv(OUT/'final_manifest.csv',index=False,encoding='utf-8-sig')
shutil.copy2(clf_path,OUT/'frozen_classifier.joblib')
shutil.copy2(reg_path,OUT/'frozen_regressor.joblib')
(OUT/'protocol.json').write_text(json.dumps({'seed':SEED,'n_general':10,'n_boundary':10,'thresholds_C':[27,30,40], 'rule':'Do not retrain, tune, or change thresholds after EnergyPlus results are revealed. Evaluate groups separately. Tmax regression applies only where EnergyPlus Tmax <40C.'},indent=2),encoding='utf-8')
print('Saved:',OUT/'final_manifest.csv')
print(plan[['Validation_ID','Holdout_Group','Predicted_Risk_Class']].to_string(index=False))
