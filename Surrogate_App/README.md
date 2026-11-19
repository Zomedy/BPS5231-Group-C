# Data Center Heatwave Resilience — frozen ML surrogate

This handoff contains the **actual frozen fitted models** used in Stage 9 (not dummy models):

- `models/classifier.joblib`: Stage 8 selected direct HistGradientBoostingClassifier, trained on original 500 + adaptive 40 EnergyPlus simulations. Predicts Safe/Warning/Failure.
- `models/regressor.joblib`: Stage 7 HistGradientBoostingRegressor trained on the subset with `Tmax_C < 40` from original 500 + adaptive 40. Predicts Data Hall zone-mean maximum air temperature (°C).

Both models use **exactly five input features, in this order**:

| API field | Valid input |
|---|---|
| `Heatwave_Intensity_C` | 0–10 °C |
| `Heatwave_Duration_day` | integer 1–5 |
| `IT_Load_Wm2` | 300–1000 W/m² |
| `Capacity_Ratio` | 0.9–1.3 |
| `Cooling_Setpoint_C` | 20–27 °C |

## Run locally (Python 3.11+ suggested)

```bash
python -m pip install -r requirements.txt
python predictor.py
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for interactive API testing.

### Request example

`POST /predict`

```json
{
  "Heatwave_Intensity_C": 6,
  "Heatwave_Duration_day": 3,
  "IT_Load_Wm2": 600,
  "Capacity_Ratio": 1.1,
  "Cooling_Setpoint_C": 24
}
```

The response includes `predicted_risk_class`, `predicted_tmax_c`, uncalibrated class probabilities, derived design/available capacity, and model limitations.

## Critical research limitations

- These models are based on a **single Singapore IWEC EnergyPlus data-center configuration**, with artificial heatwaves centered around May 21; this is not a general data-center predictor.
- Safe <27 °C, Warning 27–<30 °C, Failure >=30 °C are **provisional study labels**, not established equipment safety thresholds.
- The temperature regressor excluded cases with Tmax >=40 °C. It may output a number for a failure case, but **do not present that number as reliable in catastrophic failure conditions**.
- The classification probabilities are **not calibrated**; never present them as true real-world failure probabilities.
- Capacity formula is a **model assumption**: `Q_design_kW = 68.089 * IT_Load_Wm2 / 600`; `Q_available_kW = Q_design_kW * Capacity_Ratio * (1 - 0.02 * Heatwave_Intensity_C)`. The 2%/°C derating is an assumption, not universal equipment behavior.
- Final independent validation previously reported 85% classification accuracy over 20 selected cases; 19 non-extreme cases had Tmax MAE ~1.009 °C. These results are specific to that holdout, including intentionally selected boundary cases.
- **Never load `.joblib` from untrusted sources**: pickle-based serialization executes code at load time. These artifacts were taken from the project's own Stage 9 frozen model package.
- scikit-learn model serialization can be version-sensitive. These artifacts were loaded and tested using scikit-learn 1.8.0; use the pinned version or re-export in a controlled environment.

## Suggested App UI

Five sliders/inputs, Predict button, three-class risk result, Tmax result with caution banner when appropriate, uncalibrated probability bar chart, available cooling capacity, model assumptions panel, and a small note that inference is fast but is not a new EnergyPlus simulation. Keep API on trusted server; do not embed `.joblib` into client-side JavaScript.
