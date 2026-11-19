"""Run ONCE after Stage 9 EnergyPlus finishes. Never use final results for model selection.
Usage: python stage9_evaluate.py
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,mean_absolute_error,mean_squared_error
R=Path(__file__).resolve().parent
p=R/'stage9_final_results.csv'
d=pd.read_csv(p,encoding='utf-8-sig')
if len(d)!=20 or not d.Simulation_Status.eq('SUCCESS').all():
    raise RuntimeError('Expected 20 successful simulations; inspect stage9_final_results.csv')
out=R/'stage9_evaluation';out.mkdir(exist_ok=True)
labels=['Safe','Warning','Failure'];rows=[]
for group,sub in [('General',d[d.Holdout_Group.eq('General')]),('Boundary',d[d.Holdout_Group.eq('Boundary')]),('All',d)]:
    actual=sub.EnergyPlus_Risk_Class.astype(str);pred=sub.Predicted_Risk_Class.astype(str)
    nonrun=sub[sub.EnergyPlus_Tmax_C<40]
    row={'Group':group,'N':len(sub),'Accuracy':accuracy_score(actual,pred),
         'Macro_F1':f1_score(actual,pred,labels=labels,average='macro',zero_division=0),
         'Failure_Recall':recall_score(actual,pred,labels=['Failure'],average='macro',zero_division=0),
         'Failure_N':int((actual=='Failure').sum()),
         'Failure_as_Safe':int(((actual=='Failure')&(pred=='Safe')).sum()),
         'Operational_N':len(nonrun),'Runaway_N':int((sub.EnergyPlus_Tmax_C>=40).sum()),
         'Operational_Tmax_MAE_C':np.nan,'Operational_Tmax_RMSE_C':np.nan}
    if len(nonrun):
        a=nonrun.EnergyPlus_Tmax_C.to_numpy(float);b=nonrun.Predicted_Tmax_C_Operational_Model.to_numpy(float)
        row['Operational_Tmax_MAE_C']=mean_absolute_error(a,b)
        row['Operational_Tmax_RMSE_C']=np.sqrt(mean_squared_error(a,b))
    rows.append(row)
    cm=confusion_matrix(actual,pred,labels=labels)
    pd.DataFrame(cm,index=labels,columns=labels).rename_axis('Actual').to_csv(out/f'confusion_{group}.csv',encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(out/'final_metrics.csv',index=False,encoding='utf-8-sig')
d.to_csv(out/'final_case_predictions.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False))
print('Results:',out)
