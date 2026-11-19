from pathlib import Path
import csv
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import qmc


# ============================================================
# 0. USER CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BASELINE_IDF = BASE_DIR / "baseline.idf"
BASELINE_EPW = BASE_DIR / "SGP_SINGAPORE_IWEC.epw"

ENERGYPLUS_EXE = Path(
    r"C:/EnergyPlusV9-4-0/energyplus.exe"
)

CASES_DIR = BASE_DIR / "stage9_final_cases"
DATASET_CSV = BASE_DIR / "stage9_final_results.csv"
SAMPLES_CSV = BASE_DIR / "lhs_samples.csv"


# ----------------------------
# Number of simulations
# ----------------------------

N_CASES = 500

# Change to 500 after pilot validation.
RANDOM_SEED = 5231


# ============================================================
# 1. INPUT VARIABLE RANGES
# ============================================================

HEATWAVE_INTENSITY_RANGE = (0.0, 10.0)     # degC

# Duration is discrete integer:
HEATWAVE_DURATION_RANGE = (1, 5)           # days

IT_LOAD_RANGE = (300.0, 1000.0)            # W/m2

CAPACITY_RATIO_RANGE = (0.9, 1.3)

COOLING_SETPOINT_RANGE = (20.0, 27.0)      # degC

CAPACITY_DERATE_PER_C = 0.02

# ============================================================
# 2. FIXED BASELINE PARAMETERS
# ============================================================

BASELINE_CAPACITY_W = 68089.0

UNDERFLOOR_CAPACITY_W = 3307.0
CEILING_CAPACITY_W = 5366.0

BASELINE_IT_LOAD_WM2 = 600.0
BASELINE_SETPOINT_C = 24.0

# Baseline peak cooling demand day:
PEAK_MONTH = 5
PEAK_DAY = 21

# Evaluation period
PRE_EVENT_DAYS = 2
POST_EVENT_DAYS = 2


# ============================================================
# 3. THERMAL PERFORMANCE DEFINITIONS
# ============================================================

# Setpoint deviation:
SETPOINT_TOLERANCE_C = 1.0

# Store all three thresholds.
WARNING_TEMP_1_C = 27.0
WARNING_TEMP_2_C = 28.0
CRITICAL_TEMP_C = 30.0

# Recovery:
RECOVERY_TOLERANCE_C = 1.0

# Temperature must remain recovered for this long
RECOVERY_STABLE_HOURS = 2

# Maximum post-event observation period
RECOVERY_OBSERVATION_HOURS = (
    POST_EVENT_DAYS * 24
)


# ============================================================
# 4. FILE CHECKS
# ============================================================

for file in [
    BASELINE_IDF,
    BASELINE_EPW,
    ENERGYPLUS_EXE,
]:

    if not file.exists():

        print(f"ERROR: Missing file:\n{file}")
        sys.exit(1)


CASES_DIR.mkdir(exist_ok=True)


# ============================================================
# 5. READ MASTER IDF
# ============================================================

BASELINE_IDF_TEXT = BASELINE_IDF.read_text(
    encoding="utf-8",
    errors="ignore"
)


# ============================================================
# 6. IDF OBJECT UTILITIES
# ============================================================

def split_idf_objects(text):

    return text.split(";")


def find_exact_object(
    objects,
    object_type,
    name_contains,
):

    # 在IDF对象列表中寻找指定对象。

    # EnergyPlus IDF中经常存在这种结构：

    #     ! Computing gain
    #     OtherEquipment, BLOCK2:DATAHALL Computing gain,
    #         ...

    # 也就是说，一个object字符串的开头可能先出现注释，
    # 因此不能简单使用 startswith("OtherEquipment,")。

    # 这里的处理方法：
    # 1. 逐行读取object；
    # 2. 删除空行；
    # 3. 忽略以 ! 开头的纯注释行；
    # 4. 找到真正的EnergyPlus object第一行；
    # 5. 再判断object type和名称。

    # 这样可以避免DesignBuilder自动生成的注释影响对象识别。

    object_type_upper = object_type.upper()
    name_upper = name_contains.upper()

    for i, obj in enumerate(objects):

        # 将整个object拆成多行
        lines = obj.splitlines()

        # 保存去除注释后的有效IDF行
        valid_lines = []

        for line in lines:

            stripped = line.strip()

            # 跳过空行
            if not stripped:
                continue

            # 跳过整行注释，例如：
            # ! Computing gain
            if stripped.startswith("!"):
                continue

            # 删除行尾注释。
            #
            # 例如：
            # OtherEquipment, xxx, !- Name
            #
            # 只保留：
            # OtherEquipment, xxx,
            code_part = stripped.split("!", 1)[0].strip()

            if code_part:
                valid_lines.append(code_part)

        # 如果没有有效内容，跳过
        if not valid_lines:
            continue

        # 将有效代码重新组合
        cleaned_code = " ".join(valid_lines)

        upper = cleaned_code.upper()

        # 必须同时满足：
        #
        # 1. 真正的object类型正确
        # 2. object名称中包含指定名称
        if (
            upper.startswith(object_type_upper + ",")
            and name_upper in upper
        ):
            return i

    # 找不到则返回None
    return None


# ============================================================
# 7. MODIFY IT SERVER LOAD
# ============================================================

def modify_it_load(objects, new_it_wm2):

    index = find_exact_object(
        objects,
        "OtherEquipment",
        "BLOCK2:DATAHALL Computing gain",
    )

    if index is None:
        raise RuntimeError(
            "Data Hall Computing gain object not found."
        )

    obj = objects[index]

    pattern = (
        r"(?<![\d.])600(?:\.0+)?(?![\d.])"
    )

    replacement = f"{new_it_wm2:.4f}"

    modified, count = re.subn(
        pattern,
        replacement,
        obj,
        count=1,
    )

    if count != 1:

        raise RuntimeError(
            "Could not uniquely replace "
            "600 W/m2 IT load."
        )

    objects[index] = modified
    print(
    f"  ✓ IT Load: "
    f"{BASELINE_IT_LOAD_WM2:.1f} -> "
    f"{new_it_wm2:.3f} W/m2"
)

# ============================================================
# 8. MODIFY COOLING SETPOINT
# ============================================================

def modify_cooling_setpoint(
    objects,
    new_setpoint_c,
):

    index = find_exact_object(
        objects,
        "Schedule:Compact",
        "BLOCK2:DATAHALL Cooling SP Sch",
    )

    if index is None:

        raise RuntimeError(
            "Data Hall cooling setpoint "
            "schedule not found."
        )

    obj = objects[index]

    # The actual baseline object contains:
    # Until: 24:00, 24

    pattern = (
        r"(Until:\s*24:00\s*,\s*)"
        r"24(?:\.0+)?"
    )

    replacement = (
        r"\g<1>"
        + f"{new_setpoint_c:.4f}"
    )

    modified, count = re.subn(
        pattern,
        replacement,
        obj,
        count=1,
        flags=re.IGNORECASE,
    )

    if count != 1:

        raise RuntimeError(
            "Could not uniquely replace "
            "Data Hall cooling setpoint."
        )

    objects[index] = modified

    print(
    f"  ✓ Cooling Setpoint: "
    f"{BASELINE_SETPOINT_C:.1f} -> "
    f"{new_setpoint_c:.3f} C"
)


# ============================================================
# 9. MODIFY DATA HALL COOLING CAPACITY
# ============================================================

def modify_capacity(
    objects,
    new_capacity_w,
):

    index = find_exact_object(
        objects,
        "ZoneHVAC:IdealLoadsAirSystem",
        "BLOCK2:DATAHALL Ideal Loads Air",
    )

    if index is None:

        raise RuntimeError(
            "Data Hall Ideal Loads Air System "
            "not found."
        )

    obj = objects[index]

    pattern = (
        r"(?<![\d.])68089(?:\.0+)?(?![\d.])"
    )

    replacement = (
        f"{new_capacity_w:.0f}"
    )

    modified, count = re.subn(
        pattern,
        replacement,
        obj,
        count=1,
    )

    if count != 1:

        raise RuntimeError(
            "Could not uniquely replace "
            "68089 W cooling capacity."
        )

    objects[index] = modified

    print(
    f"  ✓ Cooling Capacity: "
    f"{BASELINE_CAPACITY_W/1000:.3f} -> "
    f"{new_capacity_w/1000:.3f} kW"
)

# ============================================================
# 10. GENERATE LATIN HYPERCUBE SAMPLES
# ============================================================

def generate_lhs_cases():

    sampler = qmc.LatinHypercube(
        d=5,
        seed=RANDOM_SEED,
    )

    sample = sampler.random(
        n=N_CASES
    )

    # Scale continuous dimensions manually.

    intensity = (
        HEATWAVE_INTENSITY_RANGE[0]
        + sample[:, 0]
        * (
            HEATWAVE_INTENSITY_RANGE[1]
            - HEATWAVE_INTENSITY_RANGE[0]
        )
    )

    # Duration: convert LHS continuous variable
    # into integer 1-5 days.
    duration = np.floor(
        sample[:, 1] * 5
    ).astype(int) + 1

    duration = np.clip(
        duration,
        1,
        5,
    )

    it_load = (
        IT_LOAD_RANGE[0]
        + sample[:, 2]
        * (
            IT_LOAD_RANGE[1]
            - IT_LOAD_RANGE[0]
        )
    )

    capacity_ratio = (
        CAPACITY_RATIO_RANGE[0]
        + sample[:, 3]
        * (
            CAPACITY_RATIO_RANGE[1]
            - CAPACITY_RATIO_RANGE[0]
        )
    )

    setpoint = (
        COOLING_SETPOINT_RANGE[0]
        + sample[:, 4]
        * (
            COOLING_SETPOINT_RANGE[1]
            - COOLING_SETPOINT_RANGE[0]
        )
    )

    rows = []

    for i in range(N_CASES):

        rows.append({

            "Case_ID":
                f"Case_{i + 1:04d}",

            "Heatwave_Intensity_C":
                round(float(intensity[i]), 3),

            "Heatwave_Duration_day":
                int(duration[i]),

            "IT_Load_Wm2":
                round(float(it_load[i]), 3),

            "Capacity_Ratio":
                round(
                    float(capacity_ratio[i]),
                    4,
                ),

            "Cooling_Setpoint_C":
                round(float(setpoint[i]), 3),
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        SAMPLES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    return df


# ============================================================
# 11. DETERMINE HEATWAVE DATES
# ============================================================

def heatwave_dates(duration):

    peak = datetime(
        2021,
        PEAK_MONTH,
        PEAK_DAY,
    )

    # Rule agreed:
    #
    # 1 day -> May 21
    # 2 days -> May 20-21
    # 3 days -> May 20-22
    # 4 days -> May 19-22
    # 5 days -> May 19-23

    days_before = duration // 2

    start = (
        peak
        - timedelta(days=days_before)
    )

    end = (
        start
        + timedelta(days=duration - 1)
    )

    evaluation_start = (
        start
        - timedelta(days=PRE_EVENT_DAYS)
    )

    evaluation_end = (
        end
        + timedelta(days=POST_EVENT_DAYS)
    )

    return (
        start,
        end,
        evaluation_start,
        evaluation_end,
    )


# ============================================================
# 12. MODIFY EPW FOR HEATWAVE
# ============================================================

def create_heatwave_epw(
    output_epw,
    intensity_c,
    duration,
):

    (
        heatwave_start,
        heatwave_end,
        _,
        _,
    ) = heatwave_dates(duration)

    lines = BASELINE_EPW.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    # EPW has 8 header lines.
    header = lines[:8]
    weather_lines = lines[8:]

    modified_lines = []

    for line in weather_lines:

        fields = line.split(",")

        if len(fields) < 7:

            modified_lines.append(line)
            continue

        try:

            month = int(fields[1])
            day = int(fields[2])

            current_date = datetime(
                2021,
                month,
                day,
            )

            if (
                heatwave_start.date()
                <= current_date.date()
                <= heatwave_end.date()
            ):

                # EPW field 7 (index 6)
                # = Dry Bulb Temperature C

                dry_bulb = float(
                    fields[6]
                )

                fields[6] = (
                    f"{dry_bulb + intensity_c:.2f}"
                )

        except (ValueError, IndexError):

            pass

        modified_lines.append(
            ",".join(fields)
        )

    output_epw.write_text(
        "\n".join(
            header + modified_lines
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        heatwave_start,
        heatwave_end,
    )


# ============================================================
# 13. RUN ENERGYPLUS
# ============================================================

def run_energyplus(
    idf_file,
    epw_file,
    output_dir,
):

    command = [
    str(ENERGYPLUS_EXE),

    "-w",
    str(epw_file),

    # --------------------------------------------------------
    # -r = 运行 ReadVarsESO
    #
    # EnergyPlus 主模拟结束后会产生 eplusout.eso，
    # 但我们的 Python 后处理程序需要读取 eplusout.csv。
    #
    # 加入 -r 后，EnergyPlus 会自动调用 ReadVarsESO，
    # 将 ESO 中的逐时结果转换成 CSV。
    # --------------------------------------------------------
    "-r",

    "-d",
    str(output_dir),

    str(idf_file),
]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    err_file = (
        output_dir / "eplusout.err"
    )

    if not err_file.exists():

        return (
            False,
            "NO_ERR_FILE",
        )

    err_text = err_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if (
        "EnergyPlus Completed Successfully"
        in err_text
    ):

        return (
            True,
            "SUCCESS",
        )

    return (
        False,
        "FAILED",
    )


# ============================================================
# 14. FIND ENERGYPLUS CSV COLUMN
# ============================================================

def find_column(
    dataframe,
    required_strings,
):

    required_strings = [
        s.upper()
        for s in required_strings
    ]

    for column in dataframe.columns:

        upper = column.upper()

        if all(
            text in upper
            for text in required_strings
        ):

            return column

    return None


# ============================================================
# 15. READ HOURLY ENERGYPLUS RESULTS
# ============================================================

def load_hourly_results(
    output_dir,
):

    csv_file = (
        output_dir / "eplusout.csv"
    )

    if not csv_file.exists():

        raise RuntimeError(
            "eplusout.csv not found."
        )

    df = pd.read_csv(csv_file)

    date_column = find_column(
        df,
        ["Date/Time"],
    )

    if date_column is None:

        # EnergyPlus normally names it
        # exactly "Date/Time"
        date_column = df.columns[0]

    return df, date_column


# ============================================================
# 16. CONVERT ENERGYPLUS DATE/TIME
# ============================================================

def parse_energyplus_datetime(
    value,
):

    text = str(value).strip()

    # Typical EnergyPlus:
    # "05/21  01:00:00"

    match = re.search(
        r"(\d{1,2})/(\d{1,2})"
        r"\s+"
        r"(\d{1,2}):(\d{2}):(\d{2})",
        text,
    )

    if not match:

        return pd.NaT

    month = int(match.group(1))
    day = int(match.group(2))
    hour = int(match.group(3))
    minute = int(match.group(4))

    # EnergyPlus may report 24:00.
    if hour == 24:

        base = datetime(
            2021,
            month,
            day,
        )

        return (
            base
            + timedelta(days=1)
        )

    return datetime(
        2021,
        month,
        day,
        hour,
        minute,
    )


# ============================================================
# 17. CALCULATE OUTPUT METRICS
# ============================================================

def calculate_metrics(
    output_dir,
    setpoint_c,
    heatwave_start,
    heatwave_end,
):

    df, date_col = (
        load_hourly_results(
            output_dir
        )
    )

    # df["ParsedDateTime"] = (
    #     df[date_col].apply(
    #         parse_energyplus_datetime
    #     )
    # )
    
    df = df.copy()

    parsed_datetime = df[date_col].apply(
        parse_energyplus_datetime
    )

        # 使用 assign 一次性加入时间列
    df = df.assign(
        ParsedDateTime=parsed_datetime
    )


    df = df.dropna(
        subset=["ParsedDateTime"]
    )


    # --------------------------------------------------------
    # Find Data Hall air temperature
    # --------------------------------------------------------

    temp_col = find_column(
        df,
        [
            "BLOCK2:DATAHALL",
            "Zone Mean Air Temperature",
        ],
    )

    if temp_col is None:

        raise RuntimeError(
            "Data Hall air temperature "
            "column not found."
        )


    # --------------------------------------------------------
    # Find Data Hall total cooling rate
    # --------------------------------------------------------

    cooling_col = find_column(
        df,
        [
            "BLOCK2:DATAHALL",
            "Zone Ideal Loads Supply Air "
            "Total Cooling Rate",
        ],
    )

    if cooling_col is None:

        raise RuntimeError(
            "Data Hall cooling rate "
            "column not found."
        )


    # --------------------------------------------------------
    # Define windows
    # --------------------------------------------------------

    hw_start = heatwave_start

    # Heatwave runs until midnight after
    # the final heatwave day.
    hw_end_exclusive = (
        heatwave_end
        + timedelta(days=1)
    )

    eval_start = (
        hw_start
        - timedelta(
            days=PRE_EVENT_DAYS
        )
    )

    eval_end_exclusive = (
        hw_end_exclusive
        + timedelta(
            days=POST_EVENT_DAYS
        )
    )


    evaluation = df[
        (
            df["ParsedDateTime"]
            >= eval_start
        )
        &
        (
            df["ParsedDateTime"]
            < eval_end_exclusive
        )
    ].copy()


    heatwave = df[
        (
            df["ParsedDateTime"]
            >= hw_start
        )
        &
        (
            df["ParsedDateTime"]
            < hw_end_exclusive
        )
    ].copy()


    post_event = df[
        (
            df["ParsedDateTime"]
            >= hw_end_exclusive
        )
        &
        (
            df["ParsedDateTime"]
            < eval_end_exclusive
        )
    ].copy()


    if evaluation.empty:

        raise RuntimeError(
            "Evaluation window is empty."
        )

    if heatwave.empty:

        raise RuntimeError(
            "Heatwave window is empty."
        )


    # --------------------------------------------------------
    # PRIMARY OUTPUT: Tmax
    # Entire evaluation window
    # --------------------------------------------------------

    tmax = float(
        evaluation[temp_col].max()
    )


    # --------------------------------------------------------
    # Peak Data Hall cooling
    # W -> kW
    # --------------------------------------------------------

    peak_cooling_kw = float(
        heatwave[cooling_col].max()
        / 1000.0
    )


    # --------------------------------------------------------
    # Heatwave Cooling Energy
    #
    # Hourly rate [W].
    # Each row represents 1 hour.
    #
    # W * h / 1000 = kWh
    # --------------------------------------------------------

    cooling_energy_kwh = float(
        heatwave[cooling_col].sum()
        / 1000.0
    )


    duration_hours = (
        len(heatwave)
    )

    if duration_hours > 0:

        average_cooling_kw = (
            cooling_energy_kwh
            / duration_hours
        )

    else:

        average_cooling_kw = np.nan


    # --------------------------------------------------------
    # Thermal performance hours
    # Hourly output => each row = 1 hour
    # --------------------------------------------------------

    setpoint_threshold = (
        setpoint_c
        + SETPOINT_TOLERANCE_C
    )

    setpoint_deviation_hours = int(
        (
            evaluation[temp_col]
            > setpoint_threshold
        ).sum()
    )

    hours_above_27 = int(
        (
            evaluation[temp_col]
            > WARNING_TEMP_1_C
        ).sum()
    )

    hours_above_28 = int(
        (
            evaluation[temp_col]
            > WARNING_TEMP_2_C
        ).sum()
    )

    hours_above_30 = int(
        (
            evaluation[temp_col]
            >= CRITICAL_TEMP_C
        ).sum()
    )


    # --------------------------------------------------------
    # Recovery time
    #
    # Recovered =
    # <= setpoint + 1C
    # continuously for >=2 hours.
    # --------------------------------------------------------

    recovery_threshold = (
        setpoint_c
        + RECOVERY_TOLERANCE_C
    )

    recovery_time_h = np.nan
    recovery_within_48h = 0

    post_event = (
        post_event
        .sort_values(
            "ParsedDateTime"
        )
        .reset_index(drop=True)
    )

    stable_needed = (
        RECOVERY_STABLE_HOURS
    )

    for i in range(
        0,
        len(post_event)
        - stable_needed
        + 1,
    ):

        segment = post_event.iloc[
            i:i + stable_needed
        ]

        if (
            segment[temp_col]
            <= recovery_threshold
        ).all():

            recovery_start = (
                segment.iloc[0][
                    "ParsedDateTime"
                ]
            )

            recovery_time_h = (
                recovery_start
                - hw_end_exclusive
            ).total_seconds() / 3600.0

            recovery_within_48h = 1

            break


    return {

        "Tmax_C":
            round(tmax, 3),

        "Peak_DataHall_Cooling_kW":
            round(
                peak_cooling_kw,
                3,
            ),

        "Heatwave_Cooling_Energy_kWh":
            round(
                cooling_energy_kwh,
                3,
            ),

        "Average_Cooling_Power_kW":
            round(
                average_cooling_kw,
                3,
            ),

        "Setpoint_Deviation_Hours":
            setpoint_deviation_hours,

        "Hours_Above_27C":
            hours_above_27,

        "Hours_Above_28C":
            hours_above_28,

        "Hours_Above_30C":
            hours_above_30,

        "Recovery_Time_h":
            (
                round(
                    recovery_time_h,
                    3,
                )
                if not np.isnan(
                    recovery_time_h
                )
                else np.nan
            ),

        "Recovery_Within_48h":
            recovery_within_48h,
    }



# ============================================================
# FINAL HOLDOUT ONLY: run 20 frozen cases, no LHS generation.
# ============================================================
MANIFEST = BASE_DIR / "stage9_plan" / "final_manifest.csv"
if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)
manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
required = ["Validation_ID", "Heatwave_Intensity_C", "Heatwave_Duration_day", "IT_Load_Wm2", "Capacity_Ratio", "Cooling_Setpoint_C"]
missing = [x for x in required if x not in manifest.columns]
if missing:
    raise ValueError(f"Missing manifest columns: {missing}")
if manifest.Validation_ID.duplicated().any():
    raise ValueError("Duplicate Validation_ID")
results = []
for n, case in manifest.iterrows():
    row = case.to_dict()
    case_id = str(row["Validation_ID"])
    print(f"\n[{n+1}/{len(manifest)}] {case_id}", flush=True)
    intensity = float(row["Heatwave_Intensity_C"])
    duration = int(row["Heatwave_Duration_day"])
    it_load = float(row["IT_Load_Wm2"])
    ratio = float(row["Capacity_Ratio"])
    setpoint = float(row["Cooling_Setpoint_C"])
    design_w = BASELINE_CAPACITY_W * it_load / BASELINE_IT_LOAD_WM2
    available_w = design_w * ratio * max(0, 1-CAPACITY_DERATE_PER_C*intensity)
    row["Calculated_Available_Capacity_kW"] = round(available_w/1000, 3)
    row["EnergyPlus_Tmax_C"] = np.nan
    row["EnergyPlus_Risk_Class"] = ""
    row["Simulation_Status"] = "NOT_RUN"
    case_dir = CASES_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    idf = case_dir / f"{case_id}.idf"
    epw = case_dir / f"{case_id}.epw"
    try:
        objects = split_idf_objects(BASELINE_IDF_TEXT)
        modify_it_load(objects, it_load)
        modify_cooling_setpoint(objects, setpoint)
        modify_capacity(objects, available_w)
        idf.write_text(";".join(objects), encoding="utf-8")
        start, end = create_heatwave_epw(epw, intensity, duration)
        success, status = run_energyplus(idf, epw, output_dir)
        row["Simulation_Status"] = status
        if success:
            metrics = calculate_metrics(output_dir, setpoint, start, end)
            row.update({"EnergyPlus_" + k: v for k,v in metrics.items()})
            tmax = float(metrics["Tmax_C"])
            row["EnergyPlus_Tmax_C"] = tmax
            row["EnergyPlus_Risk_Class"] = "Safe" if tmax < 27 else ("Warning" if tmax < 30 else "Failure")
            row["EnergyPlus_Thermal_Runaway"] = int(tmax >= 40)
            if pd.notna(row.get("Predicted_Tmax_C_Operational_Model")):
                row["Tmax_Error_Actual_Minus_Predicted_C"] = round(tmax - float(row["Predicted_Tmax_C_Operational_Model"]), 3)
            row["Risk_Class_Match"] = row["EnergyPlus_Risk_Class"] == row.get("Predicted_Risk_Class")
        print(f"  {row['Simulation_Status']}: Tmax={row['EnergyPlus_Tmax_C']}", flush=True)
    except Exception as exc:
        row["Simulation_Status"] = "ERROR"
        row["Error_Message"] = str(exc)
        print(f"  ERROR: {exc}", flush=True)
    results.append(row)
    pd.DataFrame(results).to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")
print(f"\nFinished. Results: {DATASET_CSV}")
