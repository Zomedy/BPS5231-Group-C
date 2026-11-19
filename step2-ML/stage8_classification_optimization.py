"""Stage 8: compare direct classification vs regression-based risk labels.
No EnergyPlus runs. Development-only validation cases must NOT be called final holdout.
Usage: python stage8_classification_optimization.py
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score, recall_score, mean_absolute_error
import joblib

SEED=5231
FEATURES=['Heatwave_Intensity_C','Heatwave_Duration_day','IT_Load_Wm2','Capacity_Ratio','Cooling_Setpoint_C']
LABELS=['Safe','Warning','Failure']

def risk(t):
    a=np.asarray(t,dtype=float)
    return np.where(a<27,'Safe',np.where(a<30,'Warning','Failure'))

def prepare(path,kind):
    d=pd.read_csv(path,encoding='utf-8-sig')
    d=d[d.Simulation_Status.eq('SUCCESS')].copy()
    d['Tmax_C']=pd.to_numeric(d['Tmax_C'] if kind=='original' else d['EnergyPlus_Tmax_C'],errors='coerce')
    d['Sample_ID']=d['Case_ID'].astype(str) if kind=='original' else d['Validation_ID'].astype(str)
    for f in FEATURES:d[f]=pd.to_numeric(d[f],errors='coerce')
    if d[FEATURES+['Tmax_C']].isna().any().any():raise ValueError(f'Missing numeric data in {path}')
    if not np.isfinite(d[FEATURES+['Tmax_C']].to_numpy(float)).all():raise ValueError(f'Nonfinite values in {path}')
    if d.Sample_ID.duplicated().any():raise ValueError(f'Duplicate sample IDs in {path}')
    d['Risk_Class']=risk(d.Tmax_C)
    return d

def candidates():
    return {
      'Direct_HistGradientBoosting':('classifier',HistGradientBoostingClassifier(max_iter=180,random_state=SEED)),
      'Direct_RandomForest':('classifier',RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1)),
      'Regression_HistGradientBoosting':('regressor',HistGradientBoostingRegressor(max_iter=180,l2_regularization=1.,random_state=SEED)),
      'Regression_RandomForest':('regressor',RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=SEED,n_jobs=-1))
    }

def predict(model,kind,x):
    p=model.predict(x)
    return (p if kind=='classifier' else risk(p)),(None if kind=='classifier' else p)

def scores(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return {'Accuracy':accuracy_score(y,p),'Macro_F1':f1_score(y,p,labels=LABELS,average='macro',zero_division=0),
      'Balanced_Accuracy':balanced_accuracy_score(y,p),
      'Failure_Recall':recall_score(y,p,labels=['Failure'],average='macro',zero_division=0),
      'Failure_as_Safe':int(((y=='Failure')&(p=='Safe')).sum()),'Errors':int((y!=p).sum())}

def save(df,p):df.to_csv(p,index=False,encoding='utf-8-sig')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',default='dataset.csv');ap.add_argument('--adaptive',default='adaptive_results.csv')
    ap.add_argument('--validation',default='validation_results.csv');ap.add_argument('--output',default='stage8_results')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    orig=prepare(a.original,'original');adp=prepare(a.adaptive,'adaptive');diag=prepare(a.validation,'validation')
    if (set(orig.Sample_ID)&set(adp.Sample_ID)) or (set(diag.Sample_ID)&(set(orig.Sample_ID)|set(adp.Sample_ID))):
        raise ValueError('Overlapping IDs across files')
    train=pd.concat([orig,adp],ignore_index=True)
    save(train[['Sample_ID']+FEATURES+['Tmax_C','Risk_Class']],out/'training_540.csv')
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    cv_rows=[];oof_rows=[];diag_rows=[];diag_predictions=[];frozen={}
    for name,(kind,prototype) in candidates().items():
        pred=np.empty(len(train),dtype=object)
        pred_temp=np.full(len(train),np.nan)
        fold_rows=[]
        for fold,(tr,te) in enumerate(folds.split(train[FEATURES],train.Risk_Class),1):
            model=clone(prototype)
            target=train.iloc[tr].Risk_Class if kind=='classifier' else train.iloc[tr].Tmax_C
            model.fit(train.iloc[tr][FEATURES],target)
            p,t=predict(model,kind,train.iloc[te][FEATURES]);pred[te]=p
            if t is not None:pred_temp[te]=t
            fold_rows.append({'Model':name,'Fold':fold,**scores(train.iloc[te].Risk_Class,p)})
        cv_rows.extend(fold_rows)
        for i in range(len(train)):
            oof_rows.append({'Sample_ID':train.iloc[i].Sample_ID,'Model':name,'Actual_Tmax_C':train.iloc[i].Tmax_C,
                'Actual_Class':train.iloc[i].Risk_Class,'Predicted_Class':pred[i],
                'Predicted_Tmax_C':pred_temp[i],'Correct':bool(pred[i]==train.iloc[i].Risk_Class),
                'Distance_to_Nearest_Threshold_C':min(abs(train.iloc[i].Tmax_C-27),abs(train.iloc[i].Tmax_C-30))})
        model=clone(prototype)
        model.fit(train[FEATURES],train.Risk_Class if kind=='classifier' else train.Tmax_C)
        frozen[name]={'kind':kind,'model':model,'features':FEATURES,'thresholds':[27,30],
                      'trained_on':'Original 500 + adaptive 40; NOT 12 diagnostic cases'}
        dp,dt=predict(model,kind,diag[FEATURES]);s=scores(diag.Risk_Class,dp)
        diag_rows.append({'Model':name,'Diagnostic_N':len(diag),**s,
                          'Tmax_MAE_C':(mean_absolute_error(diag.Tmax_C,dt) if dt is not None else np.nan)})
        for i in range(len(diag)):
            diag_predictions.append({'Validation_ID':diag.iloc[i].Sample_ID,'Model':name,
                'Actual_Tmax_C':diag.iloc[i].Tmax_C,'Actual_Class':diag.iloc[i].Risk_Class,
                'Predicted_Class':dp[i],'Predicted_Tmax_C':(dt[i] if dt is not None else np.nan),
                'Correct':bool(dp[i]==diag.iloc[i].Risk_Class)})
        cm=confusion_matrix(diag.Risk_Class,dp,labels=LABELS)
        save(pd.DataFrame(cm,index=LABELS,columns=LABELS).rename_axis('Actual').reset_index(),out/f'confusion_diagnostic_{name}.csv')
    cv=pd.DataFrame(cv_rows);save(cv,out/'cv_fold_scores.csv')
    summary=cv.groupby('Model',as_index=False).agg(CV_Macro_F1_Mean=('Macro_F1','mean'),CV_Macro_F1_SD=('Macro_F1','std'),
            CV_Failure_Recall_Mean=('Failure_Recall','mean'),CV_Balanced_Accuracy_Mean=('Balanced_Accuracy','mean'),
            CV_Failure_as_Safe_Total=('Failure_as_Safe','sum'))
    summary=summary.sort_values(['CV_Macro_F1_Mean','CV_Failure_Recall_Mean'],ascending=False)
    save(summary,out/'cv_model_comparison.csv')
    save(pd.DataFrame(oof_rows),out/'oof_predictions.csv')
    save(pd.DataFrame(diag_rows),out/'diagnostic_12_comparison.csv')
    save(pd.DataFrame(diag_predictions),out/'diagnostic_12_predictions.csv')
    oof=pd.DataFrame(oof_rows)
    bandrows=[]
    for name,group in oof.groupby('Model'):
        for band,subset in [('Near_threshold_1C',group[group.Distance_to_Nearest_Threshold_C<=1]),
                            ('Far_from_threshold_1C',group[group.Distance_to_Nearest_Threshold_C>1])]:
            if len(subset):
                bandrows.append({'Model':name,'Band':band,'N':len(subset),**scores(subset.Actual_Class,subset.Predicted_Class)})
    save(pd.DataFrame(bandrows),out/'threshold_error_analysis.csv')
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(summary.Model[::-1],summary.CV_Macro_F1_Mean[::-1],xerr=summary.CV_Macro_F1_SD[::-1],capsize=3)
    ax.set(xlabel='5-fold CV Macro-F1 (mean +/- SD)',title='Stage 8: direct vs regression-based classification',xlim=(0,1));fig.tight_layout()
    fig.savefig(out/'cv_model_comparison.png',dpi=180);plt.close(fig)
    # Pick model using CV only, NEVER the 12 diagnostic cases.
    winner=summary.iloc[0].Model
    joblib.dump(frozen[winner],out/'frozen_selected_classifier.joblib')
    for name,item in frozen.items():joblib.dump(item,out/f'frozen_{name}.joblib')
    note={'selected_model_by_cv_only':winner,'n_original':len(orig),'n_adaptive':len(adp),'n_diagnostic':len(diag),
      'risk_thresholds':'Safe <27, Warning 27 to <30, Failure >=30 (provisional)',
      'IMPORTANT':'12 diagnostic cases influenced adaptive sampling; NOT a final independent test. Do not select model using their scores.',
      'regression_based_warning':'Regression-based models trained on full 540, including Tmax >=40; do not interpret regression temperature predictions as validated physical extremes.',
      'next':'Freeze selected model. Generate fresh final EnergyPlus holdout without using its results to tune model.'}
    (out/'README.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print('\nCross-validation comparison:\n',summary.round(4).to_string(index=False))
    print('\nDevelopment diagnostic 12 (NOT independent):\n',pd.DataFrame(diag_rows).round(4).to_string(index=False))
    print('\nSelected by CV:',winner,'\nResults:',out.resolve())
if __name__=='__main__':main()
