"""Stage 6: choose 40 NEW EnergyPlus cases using boundary/error-focused adaptive sampling.
No EnergyPlus is executed. Requires dataset.csv, validation_results.csv.
Usage: python stage6_adaptive_sampling.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'stage6_adaptive_plan'; OUT.mkdir(exist_ok=True)
FEATURES=['Heatwave_Intensity_C','Heatwave_Duration_day','IT_Load_Wm2','Capacity_Ratio','Cooling_Setpoint_C']
LO=np.array([0,1,300,.9,20.]); HI=np.array([10,5,1000,1.3,27.]); RNG=np.random.default_rng(523106)
original=pd.read_csv(ROOT/'dataset.csv',encoding='utf-8-sig')
validation=pd.read_csv(ROOT/'validation_results.csv',encoding='utf-8-sig')
assert len(original)==500 and (original.Simulation_Status=='SUCCESS').all()
assert len(validation)==12 and (validation.Simulation_Status=='SUCCESS').all()
original['Risk_Class']=pd.cut(original.Tmax_C,[-np.inf,27,30,np.inf],right=False,labels=['Safe','Warning','Failure']).astype(str)
# Only original 500 are used for fitting. 12 independent validation cases remain untouched.
clf=HistGradientBoostingClassifier(max_iter=150,l2_regularization=1.0,random_state=5231)
clf.fit(original[FEATURES],original.Risk_Class)
reg=HistGradientBoostingRegressor(max_iter=150,l2_regularization=1.0,random_state=5231)
oper=original[original.Tmax_C<40]
reg.fit(oper[FEATURES],oper.Tmax_C)
# Pool combines neighborhoods of the 12 boundary cases and global coverage.
base=validation[FEATURES].to_numpy(float)
parts=[]
for b in base:
    z=np.tile(b,(1000,1))
    z[:,0]+=RNG.normal(0,1.15,1000)
    z[:,1]+=RNG.choice([-1,0,1],1000,p=[.15,.7,.15])
    z[:,2]+=RNG.normal(0,115,1000)
    z[:,3]+=RNG.normal(0,.035,1000)
    z[:,4]+=RNG.normal(0,1.2,1000)
    parts.append(z)
# Global samples ensure we do not only learn the previous boundary slice.
global_pool=LO+RNG.random((10000,5))*(HI-LO)
parts.append(global_pool)
pool=np.vstack(parts)
pool=np.clip(pool,LO,HI)
pool[:,1]=np.rint(pool[:,1]); pool[:,1]=np.clip(pool[:,1],1,5)
# Quantization avoids nearly identical scenarios; preserves sensible precision.
pool[:,0]=np.round(pool[:,0],2)
pool[:,2]=np.round(pool[:,2],1)
pool[:,3]=np.round(pool[:,3],3)
pool[:,4]=np.round(pool[:,4],2)
pool=np.unique(pool,axis=0)
known=np.vstack([original[FEATURES].to_numpy(float),validation[FEATURES].to_numpy(float)])
scale=HI-LO
# Remove exact and very-near duplicates of existing 512 cases.
dknown=cdist((pool-LO)/scale,(known-LO)/scale).min(axis=1)
pool=pool[dknown>.018]
probs=clf.predict_proba(pd.DataFrame(pool,columns=FEATURES))
classes=list(clf.classes_)
pfailure=probs[:,classes.index('Failure')]
# Uncertainty at three-class boundaries; plus continuous Tmax proximity to 27/30C.
uncertainty=1-probs.max(axis=1)
pred_t=reg.predict(pd.DataFrame(pool,columns=FEATURES))
near_temp=np.exp(-np.minimum(abs(pred_t-27),abs(pred_t-30))/2.0)
# Need coverage of dangerous failures, not only near 0.5 probability.
score=.65*uncertainty+.35*near_temp
# Select 30 boundary-focused and 10 coverage-focused with diversity penalty.
chosen=[]; remaining=np.ones(len(pool),dtype=bool)
for k in range(40):
    if chosen:
        diversity=cdist((pool-LO)/scale, (pool[chosen]-LO)/scale).min(axis=1)
    else:
        diversity=np.ones(len(pool))
    if k<30:
        utility=score + .22*np.minimum(diversity,.5)
    else:
        utility=.5*score + .7*np.minimum(diversity,.6)
    utility[~remaining]=-np.inf
    if chosen:
        utility[diversity<.055]=-np.inf
    idx=int(np.argmax(utility))
    if not np.isfinite(utility[idx]):
        raise RuntimeError('Not enough diverse candidates')
    chosen.append(idx); remaining[idx]=False
selected=pool[chosen]
manifest=pd.DataFrame(selected,columns=FEATURES)
manifest.insert(0,'Validation_ID',[f'ADP_{i:03d}' for i in range(1,41)])
manifest.insert(1,'Selection',['boundary_focus' if i<30 else 'coverage_focus' for i in range(40)])
manifest['Predicted_Risk_Class']=clf.predict(manifest[FEATURES])
manifest['Predicted_Failure_Score_Uncalibrated']=pfailure[chosen]
manifest['Predicted_Tmax_C_Operational_Model']=pred_t[chosen]
manifest['Expected_Available_Capacity_kW']=68.089*(manifest.IT_Load_Wm2/600)*manifest.Capacity_Ratio*(1-.02*manifest.Heatwave_Intensity_C)
manifest['Acquisition_Score']=score[chosen]
manifest.to_csv(OUT/'adaptive_manifest.csv',index=False,encoding='utf-8-sig')
(OUT/'README.txt').write_text('Stage 6 adaptive sampling: 40 new cases, 30 boundary-focused and 10 coverage-focused.\n'
 'Training uses ONLY original 500 cases; the 12 validation cases are used for choosing candidate neighborhoods, NOT fitting.\n'
 'Do not use the 12 original validation cases to train when reporting their final held-out performance.\n'
 'These 40 cases are targeted additions and cannot replace a fully independent test set.\n'
 'Predicted failure scores are UNCALIBRATED, not engineering probabilities.\n'
 'Run: python stage6_adaptive_runner.py\n',encoding='utf-8')
print('Saved:',OUT/'adaptive_manifest.csv')
print('Selected',len(manifest),'new cases. Categories:',manifest.Selection.value_counts().to_dict())
print('Existing 12 independent validation cases were NOT added to training.')
