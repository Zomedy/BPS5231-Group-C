"""Frozen EnergyPlus surrogate inference. No retraining, no EnergyPlus runtime."""
from pathlib import Path
import math
import joblib
import pandas as pd

FEATURES = ["Heatwave_Intensity_C", "Heatwave_Duration_day", "IT_Load_Wm2", "Capacity_Ratio", "Cooling_Setpoint_C"]
RANGES = {"Heatwave_Intensity_C": (0, 10), "Heatwave_Duration_day": (1, 5), "IT_Load_Wm2": (300, 1000), "Capacity_Ratio": (0.9, 1.3), "Cooling_Setpoint_C": (20, 27)}
MODEL_DIR = Path(__file__).resolve().parent / "models"

class Surrogate:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        c = joblib.load(model_dir / "classifier.joblib")
        r = joblib.load(model_dir / "regressor.joblib")
        if c.get("features") != FEATURES or r.get("features") != FEATURES:
            raise ValueError("Saved model feature order does not match deployment interface")
        if c.get("kind") != "classifier":
            raise ValueError("Unexpected classification artifact")
        self.classifier = c["model"]
        self.regressor = r["regressor"]

    def predict(self, parameters):
        if not isinstance(parameters, dict):
            raise ValueError("Expected a JSON object with five numeric fields")
        if set(parameters) != set(FEATURES):
            raise ValueError(f"Exactly these fields required: {FEATURES}")
        values = {}
        for name in FEATURES:
            raw = parameters[name]
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be numeric")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            lo, hi = RANGES[name]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}")
            values[name] = value
        if not values["Heatwave_Duration_day"].is_integer():
            raise ValueError("Heatwave_Duration_day must be an integer from 1 to 5")
        x = pd.DataFrame([{f: values[f] for f in FEATURES}], columns=FEATURES)
        risk = str(self.classifier.predict(x)[0])
        probs = self.classifier.predict_proba(x)[0]
        prob_by_class = {str(label): round(float(prob), 6) for label, prob in zip(self.classifier.classes_, probs)}
        # Regression was trained only on Tmax < 40 C. It is not valid for runaway extrapolation.
        predicted_temp = float(self.regressor.predict(x)[0])
        q_design = 68.089 * values["IT_Load_Wm2"] / 600.0
        q_available = q_design * values["Capacity_Ratio"] * (1 - 0.02 * values["Heatwave_Intensity_C"])
        warnings = ["Simulation-trained model for one Singapore data-center configuration; not a real-time sensor reading.",
                    "Risk labels use provisional thresholds: Safe <27 C, Warning 27-<30 C, Failure >=30 C.",
                    "Class probabilities are model scores, not calibrated real-world failure probabilities."]
        temperature_applicable = risk != "Failure" and predicted_temp < 40
        if not temperature_applicable:
            warnings.append("Tmax regression is trained on cases below 40 C; for predicted Failure/extreme conditions, treat Tmax as exploratory only.")
        return {
            "predicted_risk_class": risk,
            "risk_class_probabilities_uncalibrated": prob_by_class,
            "predicted_tmax_c": round(predicted_temp, 3),
            "tmax_operational_domain_caution": not temperature_applicable,
            "design_capacity_kw": round(q_design, 3),
            "available_capacity_kw": round(q_available, 3),
            "inputs": values,
            "warnings": warnings,
            "model_version": "stage9_frozen_stage8_classifier_stage7_regressor",
        }

if __name__ == "__main__":
    import json, sys
    model = Surrogate()
    data = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "Heatwave_Intensity_C": 6, "Heatwave_Duration_day": 3,
        "IT_Load_Wm2": 600, "Capacity_Ratio": 1.1, "Cooling_Setpoint_C": 24,
    }
    print(json.dumps(model.predict(data), indent=2, ensure_ascii=False))
