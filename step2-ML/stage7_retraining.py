"""Stage 7: adaptive retraining and diagnostic comparison.
Run: python stage7_retraining.py
Inputs: dataset.csv, adaptive_results.csv, validation_results.csv in same directory.
The 12 validation cases informed adaptive sampling: diagnostic, NOT independent final test.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_validate

FEATURES=['Heatwave_Intensity_C','Heatwave_Duration_day','IT_Load_Wm2','Capacity_Ratio','Cooling_Setpoint_C']
CLASSES=['Safe','Warning','Failure']
SEED=5231

def risk(t):
    return np.where(t<27,'Safe',np.where(t<30,'Warning','Failure'))

def save(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')

def models():
    return {
        'HistGradientBoosting':(HistGradientBoostingClassifier(max_iter=180,random_state=SEED),HistGradientBoostingRegressor(max_iter=180,l2_regularization=1.,random_state=SEED)),
        'RandomForest':(RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1),RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1))
    }

def read_original(path):
    x=pd.read_csv(path,encoding='utf-8-sig')
    x=x.loc[x['Simulation_Status'].eq('SUCCESS')].copy()
    x['ID']=x['Case_ID'].astype(str)
    x['Tmax_C']=pd.to_numeric(x['Tmax_C'])
    return x

def read_new(path,prefix):
    x=pd.read_csv(path,encoding='utf-8-sig')
    x=x.loc[x['Simulation_Status'].eq('SUCCESS')].copy()
    x['ID']=x['Validation_ID'].astype(str)
    x['Tmax_C']=pd.to_numeric(x['EnergyPlus_Tmax_C'])
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',default='dataset.csv')
    ap.add_argument('--adaptive',default='adaptive_results.csv')
    ap.add_argument('--validation',default='validation_results.csv')
    ap.add_argument('--output',default='stage7_results')
    a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    original=read_original(a.original)
    adaptive=read_new(a.adaptive,'ADP')
    validation=read_new(a.validation,'VAL')
    for name,frame in [('original',original),('adaptive',adaptive),('validation',validation)]:
        if frame.ID.duplicated().any():raise ValueError(f'{name}: duplicate IDs')
        if frame[FEATURES+['Tmax_C']].isna().any().any():raise ValueError(f'{name}: missing numeric data')
        if not np.isfinite(frame[FEATURES+['Tmax_C']].to_numpy(dtype=float)).all():raise ValueError(f'{name}: nonfinite values')
    if set(original.ID)&set(adaptive.ID) or set(validation.ID)&(set(original.ID)|set(adaptive.ID)):
        raise ValueError('ID overlap across datasets')
    for frame in (original,adaptive,validation):frame['Risk_Class']=risk(frame.Tmax_C.to_numpy())
    augmented=pd.concat([original,adaptive],ignore_index=True)
    save(augmented[['ID']+FEATURES+['Tmax_C','Risk_Class']],out/'augmented_dataset.csv')
    save(pd.DataFrame([{'Dataset':name,'N':len(df),'Operational_N':int((df.Tmax_C<40).sum()),'Extreme_N':int((df.Tmax_C>=40).sum()),**{f'{c}_N':int((df.Risk_Class==c).sum()) for c in CLASSES}} for name,df in [('Original',original),('Adaptive',adaptive),('Augmented',augmented),('Diagnostic_12',validation)]]),out/'dataset_counts.csv')
    cvrows=[]; evalrows=[]; predictions=[]; models_by_domain={}
    for name,(clfproto,regproto) in models().items():
        for domain,train in [('Original_500',original),('Augmented_540',augmented)]:
            clf=clfproto.__class__(**clfproto.get_params())
            reg=regproto.__class__(**regproto.get_params())
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
            cvres=cross_validate(clf,train[FEATURES],train.Risk_Class,cv=cv,scoring={'macro_f1':'f1_macro','balanced_accuracy':'balanced_accuracy'},n_jobs=1)
            cvrows.append({'Model':name,'Training':domain,'CV_Macro_F1_Mean':np.mean(cvres['test_macro_f1']),'CV_Macro_F1_SD':np.std(cvres['test_macro_f1']),'CV_Balanced_Accuracy_Mean':np.mean(cvres['test_balanced_accuracy'])})
            clf.fit(train[FEATURES],train.Risk_Class)
            regtrain=train.loc[train.Tmax_C<40]
            reg.fit(regtrain[FEATURES],regtrain.Tmax_C)
            cp=clf.predict(validation[FEATURES]);tp=reg.predict(validation[FEATURES]);actual=validation.Tmax_C.to_numpy()
            actualclass=validation.Risk_Class.to_numpy()
            evalrows.append({'Model':name,'Training':domain,'Diagnostic_N':len(validation),'Classification_Accuracy':accuracy_score(actualclass,cp),'Macro_F1':f1_score(actualclass,cp,labels=CLASSES,average='macro',zero_division=0),'Failure_Recall':recall_score(actualclass,cp,labels=['Failure'],average='macro',zero_division=0),'Tmax_MAE_C':mean_absolute_error(actual,tp),'Tmax_RMSE_C':np.sqrt(mean_squared_error(actual,tp)),'Tmax_Bias_Actual_Minus_Pred_C':np.mean(actual-tp)})
            for i,(ident,ytrue,yclass,pclass,pred) in enumerate(zip(validation.ID,actual,actualclass,cp,tp)):
                predictions.append({'Validation_ID':ident,'Model':name,'Training':domain,'Actual_Tmax_C':ytrue,'Predicted_Tmax_C':pred,'Residual_C':ytrue-pred,'Actual_Risk_Class':yclass,'Predicted_Risk_Class':pclass,'Correct_Class':bool(yclass==pclass)})
            cm=confusion_matrix(actualclass,cp,labels=CLASSES)
            save(pd.DataFrame(cm,index=CLASSES,columns=CLASSES).rename_axis('Actual').reset_index(),out/f'confusion_{name}_{domain}.csv')
            models_by_domain[(name,domain)]=(clf,reg)
    save(pd.DataFrame(cvrows),out/'classification_cv_comparison.csv')
    save(pd.DataFrame(evalrows),out/'diagnostic_12_comparison.csv')
    pred_df=pd.DataFrame(predictions);save(pred_df,out/'diagnostic_12_predictions.csv')
    fig,ax=plt.subplots(figsize=(7,5))
    for name in models():
        for domain,marker in [('Original_500','o'),('Augmented_540','s')]:
            sub=pred_df[(pred_df.Model==name)&(pred_df.Training==domain)]
            ax.scatter(sub.Actual_Tmax_C,sub.Predicted_Tmax_C,label=f'{name} {domain}',alpha=.75,marker=marker)
    ax.plot([20,40],[20,40],'k--',lw=1)
    ax.set(xlabel='EnergyPlus Tmax (C)',ylabel='Predicted Tmax (C)',title='Adaptive retraining: diagnostic cases (not final holdout)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'diagnostic_predicted_vs_actual.png',dpi=180);plt.close(fig)
    # Freeze models trained on ALL available training data. New final holdout must be generated independently.
    import joblib
    for name in models():
        clf,reg=models_by_domain[(name,'Augmented_540')]
        joblib.dump({'classifier':clf,'regressor':reg,'features':FEATURES,'classes':CLASSES,'training_rows':len(augmented),'operational_training_rows':int((augmented.Tmax_C<40).sum()),'risk_thresholds':[27,30,40]},out/f'frozen_{name}.joblib')
    note={
      'important':'The 12 validation cases influenced adaptive case selection. They are diagnostic only, NOT an independent final test.',
      'original_n':len(original),'adaptive_n':len(adaptive),'augmented_n':len(augmented),
      'recommendation':'Choose/freeze model before generating a brand-new EnergyPlus final holdout. Do not tune on that holdout.',
      'risk_thresholds':'Safe <27; Warning 27-<30; Failure >=30. Tmax regression excludes >=40.',
      'comparison_caveat':'CV scores use different training sets; diagnostic 12 are the same cases for before/after, but selection was adaptive.'}
    (out/'README.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print(pd.DataFrame(evalrows).round(3).to_string(index=False))
    print('Saved:',out.resolve())
if __name__=='__main__':main()
