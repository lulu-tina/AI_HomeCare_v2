"""長照居家照顧派單系統 - 核心運算引擎

將原始三階段流程（適配度評分 -> OR-Tools 最佳化 -> DiD 效益回溯）拆成可
重複呼叫的函式，並把所有具派單政策意義的係數集中到 PipelineConfig，供
CLI (ai_caregiver_pipeline.py) 與網頁儀表板 (app.py) 共用同一份邏輯。

另包含 BA 服務代碼併報法規防呆檢核、長照申報點數與居服員拆帳薪資試算、
星期／可服務時段／請假防呆的硬性條件，以及依日期逐日批次派單的輔助函式。
新增的欄位需求皆以 `.get()`／`pd.notna()` 方式取值，僅在資料表實際具備對應
欄位時才生效，因此仍可原封不動套用在既有的單日排班資料表格式上。
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from ortools.linear_solver import pywraplp

DEFAULT_EXCEL_PATH = "./00_DB/AI_Caregiver_Allocation_Ultimate_Database.xlsx"

# OSRM 公用路網 API 設定：預設查詢騎乘(機車/腳踏車)路網時間，逾時即降級為概算公式。
OSRM_HOST = "http://router.project-osrm.org"
OSRM_PROFILE = "biking"
OSRM_TIMEOUT_SECONDS = 3.0
OSRM_TABLE_TIMEOUT_SECONDS = 10.0
OSRM_TABLE_MAX_COORDS = 90  # 單次 /table 批次查詢座標數上限，避免超出公用伺服器限制

# Google Routes API
GOOGLE_ROUTES_URL = (
    "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
)

GOOGLE_ROUTES_TIMEOUT_SECONDS = 15.0

# 交通方式 → Google Routes travelMode
GOOGLE_TRAVEL_MODE_MAP = {
    "機車": "TWO_WHEELER",
    "大眾運輸": "TRANSIT",
}

# 服務項目強度加權係數：依體力耗費強度分級，用於疲勞度模型。
SERVICE_INTENSITY_WEIGHT = {
    "重度移位": 1.5,  # 重度移位／肢體關節活動
    "餐食管灌": 1.2,  # 餐食照顧／管灌洗頭
    "一般照護": 0.8,  # 一般家務／陪伴看護
}

# 環境排斥條件對照表 (居服員排斥 -> 案家環境)，兩側皆為 Excel 原始字串，需完全相符
EXCLUSION_MAP = {
    "拒爬高樓(無電梯)": "傳統公寓無電梯",
    "拒寵物環境": "有養寵物",
    "拒菸害環境": "有抽菸",
}

# 法定 20 小時特照培訓：案家需求類別 -> 合格居服員「核心專長證照」集合。
# 居服員若不具備對應證照，該配對於 Phase 1 直接硬性剔除，不得派單。
SPECIAL_CERT_REQUIREMENTS = {
    "失智引導與精神陪伴": {"失智症照顧專長", "精神疾病照顧專長"},
}

# 星期中文名稱 -> ISO 星期數字（1=一 ... 7=日），供「可排班星期」防呆使用。
WEEKDAY_NAME_TO_NUM = {
    "星期一": 1, "星期二": 2, "星期三": 3, "星期四": 4,
    "星期五": 5, "星期六": 6, "星期日": 7,
}
WEEKDAY_NUM_TO_NAME = {v: k for k, v in WEEKDAY_NAME_TO_NUM.items()}


def get_weekday_name(date_value) -> str:
    """將日期值轉換為中文星期名稱（"星期一"...），供派單結果表格顯示用。

    無法解析（空值、格式不符）時回傳空字串，而非拋出例外，避免單一列的日期
    格式問題導致整張結果表格無法呈現。
    """
    if pd.isna(date_value) or str(date_value).strip() == "":
        return ""
    try:
        parsed = pd.to_datetime(date_value)
    except (ValueError, TypeError):
        return ""
    return WEEKDAY_NUM_TO_NAME.get(parsed.isoweekday(), "")

# 台灣長照 2.0 常用 BA 服務代碼點數（點值，機構申報營收以此試算；1 點通常對應約 1 元）。
BA_UNIT_POINTS = {
    "BA01": 175,
    "BA02": 210,
    "BA03": 150,
    "BA04": 180,
    "BA05": 200,
    "BA07": 250,
    "BA08": 300,
    "BA09": 160,
    "BA10": 190,
}

# 無法由 Service_Code_1/2 判斷點數時（例如資料表未升級至含 BA 碼申報欄位），
# 退回以服務歷時概算點數：每 30 分鐘 = 150 點。
_FALLBACK_POINTS_PER_30MIN = 150.0


@dataclass
class PipelineConfig:
    """所有可調整的派單政策參數，預設值與原始腳本一致。"""

    # 轉場緩衝與交通時間模型
    buffer_mins: float = 15.0
    travel_min_per_km: float = 3.0

    # 勞動合規：累計連續工作達 continuous_work_limit_mins 分鐘，強制要求下一段任務
    # 與前段之間至少間隔 mandatory_break_mins 分鐘（見 Phase 2 的 _build_break_constraints）。
    continuous_work_limit_mins: float = 240.0
    mandatory_break_mins: float = 30.0

    # Phase 1：軟性適配度評分
    base_score: float = 60.0
    cert_bonus_dementia: float = 15.0
    cert_bonus_other: float = 10.0
    preferred_caregiver_bonus: float = 60.0
    continuity_performance_bonus: float = 15.0
    continuity_satisfaction_threshold: float = 4.3
    satisfaction_baseline: float = 4.0
    satisfaction_weight: float = 10.0
    travel_penalty_weight: float = 2.0
    travel_penalty_cap: float = 25.0
    # 工作負荷平衡：若資料未提供「當月可排總工時」，預設以 160 小時作為月可用工時基準
    fatigue_reference_hours: float = 160.0
    fatigue_weight: float = 10.0

    # Phase 1：車程上限硬性條件（None = 停用，不做硬性剔除）。
    # 照護連續性優先於車程限制：一般候選人受 max_travel_minutes 限制，
    # 但歷史首選居服員改用較寬鬆的 preferred_caregiver_max_travel_minutes
    # （或直接不受限，若該欄位亦為 None），避免熟悉的居服員被車程微幅超標而剔除。
    max_travel_minutes: Optional[float] = None
    preferred_caregiver_max_travel_minutes: Optional[float] = None

    # Phase 2：OR-Tools 目標函數權重
    urgent_priority_bonus: float = 50.0
    normal_priority_bonus: float = 20.0

    # 財務試算：長照申報點數換算居服員薪資的拆帳比例；
    # 僅供結果顯示與財務 KPI 試算，不參與派單決策。
    caregiver_salary_rate_per_point: float = 0.65


# ==========================================
# 資料載入
# ==========================================
def load_data(excel_path: str = DEFAULT_EXCEL_PATH, tasks_sheet_name: str = "Today_Pending_Tasks"):
    """讀取派單資料庫四張工作表。

    tasks_sheet_name 預設為現行單日排班格式的「Today_Pending_Tasks」；若改用含
    星期／日期欄位的月批次排班資料表，呼叫端可傳入 "Monthly_Pending_Tasks"。
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到檔案 {excel_path}，請確認檔案與腳本在同一目錄下。")

    df_cg = pd.read_excel(excel_path, sheet_name="Caregiver_Profiles")
    df_cl = pd.read_excel(excel_path, sheet_name="Client_Profiles")
    df_tasks = pd.read_excel(excel_path, sheet_name=tasks_sheet_name)
    df_hist = pd.read_excel(excel_path, sheet_name="Historical_Service_Logs")

    tasks = df_tasks.merge(df_cl, on="案家ID", how="left")
    return df_cg, df_cl, df_tasks, df_hist, tasks


def _get_task_field(row, base_col_name: str):
    """依序嘗試 base_col_name、base_col_name_x、base_col_name_y，取得第一個有值的欄位。

    新版月批次任務資料表的 Service_Code_1/2、Units_1/2 欄位，若在 Client_Profiles
    亦重複定義同名欄位，經 `tasks_df.merge(df_cl, on="案家ID")` 合併後，pandas 會
    自動將同名欄位改為 base_col_name_x（左表／任務本身）、base_col_name_y（右表／
    案家）。本函式確保無論合併後欄位是否被加上後綴，皆優先取任務本身、其次取
    案家層級的對應值，而不會因欄位改名誤判為「缺少代碼」。row 可為 Series 或 dict。
    """
    for col in (base_col_name, f"{base_col_name}_x", f"{base_col_name}_y"):
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return value
    return None

def apply_service_duration(
    tasks_df: pd.DataFrame,
    service_code_df: pd.DataFrame,
) -> pd.DataFrame:
    """依 Service_Code Master 的分鐘數，重算每筆任務服務歷時。"""

    result = tasks_df.copy()

    required_columns = [
        "系統代碼",
        "CareFlow排班分鐘(暫定)",
        "是否納入CareFlow",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in service_code_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Service_Code 缺少必要欄位："
            + "、".join(missing_columns)
        )

    master = service_code_df.copy()
    master = master.dropna(subset=["系統代碼"])
    master["系統代碼"] = (
        master["系統代碼"].astype(str).str.strip()
    )

    duplicated_codes = master[
        master["系統代碼"].duplicated(keep=False)
    ]["系統代碼"].unique()

    if len(duplicated_codes) > 0:
        raise ValueError(
            "Service_Code Master 有重複代碼："
            + "、".join(duplicated_codes)
        )

    master = master.set_index("系統代碼")

    calculated_minutes = []
    calculation_details = []

    for _, row in result.iterrows():
        task_id = row.get("任務ID", "未知任務")
        total_minutes = 0.0
        details = []

        for index in (1, 2):
            code_raw = row.get(f"Service_Code_{index}")
            units_raw = row.get(f"Units_{index}")

            if pd.isna(code_raw) or str(code_raw).strip() == "":
                continue

            code = str(code_raw).strip()

            if pd.isna(units_raw):
                raise ValueError(
                    f"任務 {task_id} 的 {code} 未填寫 Units"
                )

            try:
                units = float(units_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"任務 {task_id} 的 {code} Units 不是有效數字"
                )

            if units < 0:
                raise ValueError(
                    f"任務 {task_id} 的 {code} Units 不可小於 0"
                )

            if units == 0:
                continue

            if code not in master.index:
                raise ValueError(
                    f"任務 {task_id} 的服務碼 {code} "
                    "不存在於 Service_Code Master"
                )

            master_row = master.loc[code]
            status = str(
                master_row["是否納入CareFlow"]
            ).strip()

            # AA07～AA11是附加碼，不另增加服務分鐘
            if code.startswith("AA"):
                details.append(f"{code}×{units:g}=0")
                continue

            if status in {"否", "否/另模組"}:
                raise ValueError(
                    f"任務 {task_id} 使用目前不支援排班的服務碼 {code}"
                )

            minutes = master_row["CareFlow排班分鐘(暫定)"]

            if pd.isna(minutes):
                raise ValueError(
                    f"Service_Code Master 的 {code} "
                    "尚未設定CareFlow排班分鐘"
                )

            subtotal = float(minutes) * units
            total_minutes += subtotal
            details.append(
                f"{code}×{units:g}={subtotal:g}分鐘"
            )

        if total_minutes <= 0:
            raise ValueError(
                f"任務 {task_id} 無法計算出有效服務歷時"
            )

        calculated_minutes.append(total_minutes)
        calculation_details.append(" + ".join(details))

    # 保留Excel原有值，方便比較
    if "服務歷時(分鐘)" in result.columns:
        result["原服務歷時(分鐘)"] = result["服務歷時(分鐘)"]

    result["服務歷時(分鐘)"] = calculated_minutes
    result["服務歷時計算明細"] = calculation_details

    return result
# ==========================================
# 申報法規防呆：BA 服務代碼併報合規檢核
# ==========================================
def check_ba_code_compatibility(task_row) -> Optional[str]:
    """檢查單一任務的 BA 服務代碼併報合規性（依長照給付支付基準併報規則）。

    task_row 可為 DataFrame 的一列（Series）或 dict，僅需能以 `.get()` 取出
    Service_Code_1 / Service_Code_2（或合併後帶 _x/_y 後綴的同名欄位，見
    `_get_task_field`）。回傳 None 代表未發現已知違規；否則回傳違規說明文字。
    本函式僅檢核並回報，不會自動剔除或修改任務——是否略過警告並放行進入排程，
    由呼叫端（例如居督於 app.py 的人工複核介面）決定。
    """
    code1_raw = _get_task_field(task_row, "Service_Code_1")
    code2_raw = _get_task_field(task_row, "Service_Code_2")
    code1 = str(code1_raw).strip() if code1_raw is not None else ""
    code2 = str(code2_raw).strip() if code2_raw is not None else ""
    codes = {c for c in (code1, code2) if c}

    if "BA01" in codes and ("BA07" in codes or "BA23" in codes):
        return "違規：BA01（基本身體清潔）不可與 BA07/BA23（沐浴/洗頭）同時段併申報"

    if "BA02" in codes and len(codes) > 1:
        other_codes = codes - {"BA02"}
        if not other_codes.issubset({"BA22"}):
            return f"違規：BA02（基本日常照顧）除 BA22 外不得與其他項目 ({other_codes}) 評定併用"

    if "BA01" in codes and "BA24" in codes:
        return "注意：BA01 與 BA24 同時段申報可能涉及排泄項目重複，請確認計畫書核定內容"

    return None


def validate_ba_codes(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """對整批任務套用 check_ba_code_compatibility，回傳新增檢核欄位的副本。

    新增「BA代碼檢核異常」（違規說明文字或 None）與「含違規代碼」（布林值）兩欄，
    供派單前的法規防呆健檢（例如上傳資料後的即時檢核儀表板）使用；不修改傳入的
    DataFrame，亦不會排除任何任務列。
    """
    result = tasks_df.copy()
    messages = [check_ba_code_compatibility(row) for _, row in result.iterrows()]
    result["BA代碼檢核異常"] = messages
    result["含違規代碼"] = [m is not None for m in messages]
    return result


# ==========================================
# 財務試算：長照申報點數（營收）與居服員拆帳薪資
# ==========================================
def calculate_task_revenue_and_salary(tasks_df: pd.DataFrame, config: "PipelineConfig") -> pd.DataFrame:
    """試算每筆任務的長照申報點數（機構營收）與居服員拆帳薪資。

    優先以 Service_Code_1/2 對照 BA_UNIT_POINTS 點數表 × Units_1/2 計算；若任務
    缺乏服務代碼欄位（例如尚未升級至含 BA 碼申報欄位的資料表），退回以服務歷時
    概算點數，確保任何資料版本皆能得到合理的營收估計。回傳新增兩欄位的副本，
    不修改傳入的 DataFrame。
    """
    result = tasks_df.copy()
    revenues = []
    for _, row in result.iterrows():
        rev = 0.0
        for code_col, units_col in (("Service_Code_1", "Units_1"), ("Service_Code_2", "Units_2")):
            code_raw = _get_task_field(row, code_col)
            units_raw = _get_task_field(row, units_col)
            code = str(code_raw).strip() if code_raw is not None else ""
            units = float(units_raw) if units_raw is not None else 0.0
            if code in BA_UNIT_POINTS:
                rev += BA_UNIT_POINTS[code] * units

        if rev == 0.0:
            duration_raw = row.get("服務歷時(分鐘)")
            duration_mins = float(duration_raw) if pd.notna(duration_raw) else 90.0
            rev = (duration_mins / 30.0) * _FALLBACK_POINTS_PER_30MIN

        revenues.append(rev)

    result["預估長照申報點數(營收)"] = [round(r, 1) for r in revenues]
    result["預估居服員拆帳薪資"] = [round(r * config.caregiver_salary_rate_per_point, 1) for r in revenues]
    return result


# ==========================================
# 共用工具
# ==========================================
def calc_distance_km(lat1, lon1, lat2, lon2):
    """以台灣緯度估算直線距離 km（1度緯度約111km，1度經度約101km）。"""
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 101.0
    return np.sqrt(dlat**2 + dlon**2)


# 字典快取已查詢過的經緯度對 -> 路網時間(分鐘)，Phase 1 / Phase 2 共用同一份快取，
# 避免同一對座標於不同階段重複發送 API 請求。
_OSRM_TRAVEL_TIME_CACHE: dict = {}


def _round_coord(v: float) -> float:
    return round(float(v), 6)


def _osrm_fallback_minutes(lat1, lon1, lat2, lon2, travel_min_per_km, reason) -> float:
    """降級為 Haversine/歐式距離估算，並印出 Warning Log。"""
    print(
        f"[OSRM Warning] 路網查詢失敗 ({lat1},{lon1}) -> ({lat2},{lon2})，"
        f"降級為 Haversine/歐式距離估算: {reason}"
    )
    return calc_distance_km(lat1, lon1, lat2, lon2) * travel_min_per_km


def get_osrm_travel_time(lat1, lon1, lat2, lon2, travel_min_per_km=3.0):
    """查詢 OSRM 公用路網 API，回傳兩點間真實路網騎乘時間（分鐘）。

    以字典快取已查詢過的經緯度對，避免重複發送 API 請求（大量座標對可先呼叫
    `prefetch_osrm_travel_times` 以單次 /table 批次請求暖身快取）。若 API 逾時、
    網路斷線或回傳失敗，自動降級為 calc_distance_km 的概算距離公式，並印出
    Warning Log，確保 Phase 1 / Phase 2 的排程運算不因外部服務中斷而失敗。
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    key = (_round_coord(lat1), _round_coord(lon1), _round_coord(lat2), _round_coord(lon2))
    if key in _OSRM_TRAVEL_TIME_CACHE:
        return _OSRM_TRAVEL_TIME_CACHE[key]

    url = f"{OSRM_HOST}/route/v1/{OSRM_PROFILE}/{lon1},{lat1};{lon2},{lat2}"
    try:
        resp = requests.get(url, params={"overview": "false"}, timeout=OSRM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            raise ValueError(f"OSRM 回傳異常狀態: {data.get('code')}")
        travel_min = data["routes"][0]["duration"] / 60.0
    except Exception as exc:
        travel_min = _osrm_fallback_minutes(lat1, lon1, lat2, lon2, travel_min_per_km, exc)

    _OSRM_TRAVEL_TIME_CACHE[key] = travel_min
    return travel_min

def _get_google_maps_api_key():
    """
    優先從 Streamlit secrets 讀取；
    本機開發時可改由環境變數 GOOGLE_MAPS_API_KEY 提供。
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st
        return st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        return None


_GOOGLE_TRAVEL_TIME_CACHE = {}


def _parse_google_duration(duration_str):
    """
    Google duration 格式例如 '723s'、'723.5s'
    → 回傳分鐘。
    """
    if not duration_str:
        return None

    seconds = float(str(duration_str).rstrip("s"))
    return seconds / 60.0


def get_google_travel_time(
    lat1,
    lon1,
    lat2,
    lon2,
    transport_mode,
    travel_min_per_km=3.0,
):
    """
    使用 Google Routes API 計算兩點交通時間。

    transport_mode:
        機車       -> TWO_WHEELER
        大眾運輸   -> TRANSIT
    """

    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    google_mode = GOOGLE_TRAVEL_MODE_MAP.get(transport_mode)

    if google_mode is None:
        raise ValueError(
            f"不支援的常用交通工具：{transport_mode}"
        )

    key = (
        _round_coord(lat1),
        _round_coord(lon1),
        _round_coord(lat2),
        _round_coord(lon2),
        google_mode,
    )

    if key in _GOOGLE_TRAVEL_TIME_CACHE:
        return _GOOGLE_TRAVEL_TIME_CACHE[key]

    api_key = _get_google_maps_api_key()

    # 尚未設定 Google API 時，Prototype 暫時沿用 OSRM / 距離概算
    if not api_key:
        print(
            "[Google Routes Warning] 尚未設定 GOOGLE_MAPS_API_KEY，"
            "暫時使用原 OSRM 交通時間。"
        )

        return get_osrm_travel_time(
            lat1,
            lon1,
            lat2,
            lon2,
            travel_min_per_km,
        )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "originIndex,destinationIndex,"
            "status,condition,duration,distanceMeters"
        ),
    }

    body = {
        "origins": [
            {
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": float(lat1),
                            "longitude": float(lon1),
                        }
                    }
                }
            }
        ],
        "destinations": [
            {
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": float(lat2),
                            "longitude": float(lon2),
                        }
                    }
                }
            }
        ],
        "travelMode": google_mode,
        "languageCode": "zh-TW",
        "regionCode": "TW",
    }

    # routingPreference 只能用於 DRIVE / TWO_WHEELER，
    # TRANSIT 不可帶這個欄位。
    if google_mode == "TWO_WHEELER":
        body["routingPreference"] = "TRAFFIC_AWARE"

    try:
        response = requests.post(
            GOOGLE_ROUTES_URL,
            headers=headers,
            json=body,
            timeout=GOOGLE_ROUTES_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            raise ValueError("Google Routes API 未回傳路徑")

        element = data[0]

        if element.get("condition") != "ROUTE_EXISTS":
            raise ValueError(
                f"Google Routes 無可用路徑："
                f"{element.get('condition')}"
            )

        travel_min = _parse_google_duration(
            element.get("duration")
        )

        if travel_min is None:
            raise ValueError("Google Routes 未回傳 duration")

    except Exception as exc:

        print(
            f"[Google Routes Warning] "
            f"{transport_mode} 路徑查詢失敗：{exc}"
        )

        # Google 暫時失效時，不讓整個排班系統掛掉
        travel_min = get_osrm_travel_time(
            lat1,
            lon1,
            lat2,
            lon2,
            travel_min_per_km,
        )

    _GOOGLE_TRAVEL_TIME_CACHE[key] = travel_min

    return travel_min

def _fetch_osrm_table_chunk(origins, destinations, travel_min_per_km):
    """對一批 origin x destination 座標呼叫 OSRM /table 矩陣 API，一次查詢多組配對的
    路網時間，寫入共用快取。origins / destinations 皆為已四捨五入的 (lat, lon) tuple 清單。
    """
    coords = origins + destinations
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    sources = ";".join(str(i) for i in range(len(origins)))
    dest_indices = ";".join(str(len(origins) + i) for i in range(len(destinations)))
    url = f"{OSRM_HOST}/table/v1/{OSRM_PROFILE}/{coord_str}"

    try:
        resp = requests.get(
            url,
            params={"annotations": "duration", "sources": sources, "destinations": dest_indices},
            timeout=OSRM_TABLE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM /table 回傳異常狀態: {data.get('code')}")
        durations = data["durations"]
    except Exception as exc:
        print(
            f"[OSRM Warning] /table 批次路網查詢失敗（{len(origins)}x{len(destinations)} 座標對），"
            f"個別配對將於逐筆查詢時各自降級為概算公式: {exc}"
        )
        return

    for i, (o_lat, o_lon) in enumerate(origins):
        for j, (d_lat, d_lon) in enumerate(destinations):
            key = (o_lat, o_lon, d_lat, d_lon)
            duration_sec = durations[i][j]
            if duration_sec is None:
                _OSRM_TRAVEL_TIME_CACHE[key] = _osrm_fallback_minutes(
                    o_lat, o_lon, d_lat, d_lon, travel_min_per_km, "OSRM /table 無法規劃該路徑"
                )
            else:
                _OSRM_TRAVEL_TIME_CACHE[key] = duration_sec / 60.0


def prefetch_osrm_travel_times(origins, destinations, travel_min_per_km=3.0) -> None:
    """批次暖身 OSRM 路網時間快取，大幅降低 Phase 1 / Phase 2 的 API 呼叫次數。

    origins / destinations 為 (lat, lon) 的可迭代物件（例如所有居服員住家座標 x 所有
    任務地點座標）。以 OSRM `/table` 矩陣 API 一次查詢整批配對，取代逐筆呼叫
    `get_osrm_travel_time` 各自發送一次 HTTP 請求；查詢結果寫入與 `get_osrm_travel_time`
    共用的字典快取，之後逐筆呼叫即直接命中快取。任一批次查詢失敗僅印出 Warning Log
    並跳過，未快取的配對會在後續逐筆查詢時各自降級為概算公式，不影響排程結果正確性。
    """
    unique_origins = sorted({(_round_coord(lat), _round_coord(lon)) for lat, lon in origins})
    unique_destinations = sorted({(_round_coord(lat), _round_coord(lon)) for lat, lon in destinations})

    pending_origins = [
        o
        for o in unique_origins
        if any((o[0], o[1], d[0], d[1]) not in _OSRM_TRAVEL_TIME_CACHE for d in unique_destinations)
    ]
    if not pending_origins or not unique_destinations:
        return

    dest_chunk_size = max(1, OSRM_TABLE_MAX_COORDS - len(pending_origins))
    for i in range(0, len(unique_destinations), dest_chunk_size):
        dest_chunk = unique_destinations[i : i + dest_chunk_size]
        _fetch_osrm_table_chunk(pending_origins, dest_chunk, travel_min_per_km)


def calc_travel_minutes(
    lat1,
    lon1,
    lat2,
    lon2,
    config: "PipelineConfig",
    transport_mode="機車",
) -> float:
    """
    兩點間轉場時間（分鐘，不含轉場緩衝）。

    依居服員「常用交通工具」選擇 Google Routes travel mode：
    機車 -> TWO_WHEELER
    大眾運輸 -> TRANSIT
    """

    return get_google_travel_time(
        lat1,
        lon1,
        lat2,
        lon2,
        transport_mode,
        config.travel_min_per_km,
    )


def get_service_intensity_weight(task) -> float:
    """依服務項目之體力耗費強度，回傳該任務對應的疲勞度加權係數。"""
    if task["需重度移位協助(0/1)"] == 1:
        return SERVICE_INTENSITY_WEIGHT["重度移位"]
    if task["特殊照護需求"] in ("餐食照顧/管灌", "管路安全與特殊日常照護"):
        return SERVICE_INTENSITY_WEIGHT["餐食管灌"]
    return SERVICE_INTENSITY_WEIGHT["一般照護"]


def parse_time(time_str):
    if pd.isna(time_str) or time_str == "無既定行程":
        return None, None
    parts = str(time_str).split("-")
    return (
        datetime.strptime(parts[0].strip(), "%H:%M"),
        datetime.strptime(parts[1].strip(), "%H:%M"),
    )


def _check_hard_constraints(
    task,
    cg,
    config: "PipelineConfig",
    travel_time_min: Optional[float] = None,
    is_preferred_caregiver: bool = False,
):
    """檢查單一 (任務, 居服員) 配對的硬性條件。全數通過回傳 None，否則回傳未通過原因。

    抽成獨立函式供 Phase 1 過濾與「原首選替換原因」診斷共用同一套判斷邏輯。

    星期對應／時間窗涵蓋／請假排除三項檢查，僅在 task／cg 實際具備對應欄位
    （可排班星期、每日可服務時段_起/迄、請假或不排班日期、星期、日期）時才生效
    ——以 `.get()` 取值，取不到（欄位不存在或為空）即直接跳過該檢查，因此舊版
    僅有「今日既定行程」欄位、不含這些欄位的資料表行為完全不受影響。
    車程上限檢查同理，僅在呼叫端提供 travel_time_min 且 config.max_travel_minutes
    已設定（非 None）時才生效，預設為停用。
    """
    # 以下訊息刻意採「無主語」寫法（不寫「居服員」/「原居服員」），因為本函式同時
    # 供 Phase 1 一般候選人過濾（訊息僅供 reason_counts 內部除錯彙總）與
    # `_diagnose_caregiver_change` 診斷「原首選居服員」共用；後者會在回傳訊息前
    # 明確加上 `原首選居服員[ID]` 主語，若訊息本身也帶主語詞，會讓居督誤以為
    # 訊息在描述「獲派居服員」而非「原首選居服員」，見任務一問題分析。
    req_gender = task["指定居服員性別"]
    if req_gender == "限女性" and cg["性別"] != "女":
        return "案家指定女性居服員，性別不符"
    if req_gender == "限男性" and cg["性別"] != "男":
        return "案家指定男性居服員，性別不符"

    if task["需重度移位協助(0/1)"] == 1 and cg["具備重度移位體力(0/1)"] == 0:
        return "案家需重度移位協助，不具備相關體力條件"

    cg_excl = cg["特殊排斥條件"]
    if cg_excl in EXCLUSION_MAP and task["案家環境特徵"] == EXCLUSION_MAP[cg_excl]:
        return f"排斥「{EXCLUSION_MAP[cg_excl]}」環境條件"

    # 星期對應檢查 (Day-of-Week Matching)：僅當「星期」／「可排班星期」欄位確實存在於
    # 資料表中時才生效——用 `in task.index`／`in cg.index` 判斷「欄位是否存在」，而非
    # 用 `pd.notna()` 判斷「儲存格是否為空」。這兩者過去被混為一談：欄位整體不存在
    # （舊版資料表，理應跳過此檢查，維持向下相容）與欄位存在但「此列」資料缺漏或格式
    # 無法解析（新版資料表的資料品質問題）都會落入同一個 if 分支被直接跳過，導致
    # 可排班星期留空、或星期名稱格式不符的居服員被silently當作「全天候可排班」而通過
    # 硬性限制——這正是「居服員當日不在可排班星期名單內，系統卻仍將其派單」的根因。
    # 欄位存在時，資料缺漏或無法解析一律保守判定為不通過（fail-closed）並印出警告，
    # 而非靜默放行（fail-open）。
    has_weekday_cols = "星期" in task.index and "可排班星期" in cg.index
    if has_weekday_cols:
        task_weekday = task.get("星期")
        allowed_days_str = cg.get("可排班星期")
        cg_id_for_log = cg.get("居服員ID", "?")
        task_id_for_log = task.get("任務ID", "?")
        if pd.isna(task_weekday) or pd.isna(allowed_days_str) or str(allowed_days_str).strip() == "":
            print(
                f"[Hard Constraint Warning] 任務 {task_id_for_log} 或居服員 {cg_id_for_log} "
                f"的「星期」／「可排班星期」欄位資料缺漏，保守判定當日不可派單。"
            )
            return "星期或可排班星期資料缺漏，保守判定當日不可派單"

        task_wd_num = WEEKDAY_NAME_TO_NUM.get(str(task_weekday).strip())
        if task_wd_num is None:
            print(
                f"[Hard Constraint Warning] 任務 {task_id_for_log} 的「星期」欄位值"
                f"「{task_weekday}」無法辨識，保守判定當日不可派單。"
            )
            return "任務星期欄位格式無法辨識，保守判定當日不可派單"

        allowed_days = [int(d.strip()) for d in str(allowed_days_str).split(",") if d.strip().isdigit()]
        if not allowed_days:
            print(
                f"[Hard Constraint Warning] 居服員 {cg_id_for_log} 的「可排班星期」欄位值"
                f"「{allowed_days_str}」無法解析出任何星期，保守判定當日不可派單。"
            )
            return "可排班星期欄位格式無法解析，保守判定當日不可派單"

        if task_wd_num not in allowed_days:
            return "當日不排班（不在可排班星期名單）"

    # 時間窗涵蓋檢查 (Time Window Overlap)：僅當居服員有每日可服務時段欄位時生效
    cg_start_str = cg.get("每日可服務時段_起")
    cg_end_str = cg.get("每日可服務時段_迄")
    if pd.notna(cg_start_str) and pd.notna(cg_end_str):
        t_start = datetime.strptime(str(task["時間窗_開始"]).strip(), "%H:%M")
        t_end = datetime.strptime(str(task["時間窗_結束"]).strip(), "%H:%M")
        c_start = datetime.strptime(str(cg_start_str).strip(), "%H:%M")
        c_end = datetime.strptime(str(cg_end_str).strip(), "%H:%M")
        if not (t_start >= c_start and t_end <= c_end):
            return "任務時段超出每日可服務時段範圍"

    # 請假或不排班日期排除 (Leave Exclusion)：僅當任務有「日期」、居服員有請假日期欄位時生效
    leave_dates_str = cg.get("請假或不排班日期")
    task_date = task.get("日期")
    if pd.notna(leave_dates_str) and pd.notna(task_date) and str(leave_dates_str).strip() not in ("", "無"):
        leave_list = [d.strip() for d in str(leave_dates_str).split(",") if d.strip()]
        task_date_str = str(task_date).split(" ")[0]
        if task_date_str in leave_list:
            return "當日已有請假或不排班記錄"

    task_duration_hrs = task["服務歷時(分鐘)"] / 60.0
    if cg.get("今日已佔用工時(小時)", 0.0) + task_duration_hrs > cg["每日工時上限(小時)"]:
        return "今日工時已達每日上限"

    # 車程上限，惟「照護連續性」優先於「車程限制」：一般候選人受 max_travel_minutes
    # 限制（None = 不設限），但歷史首選居服員改採獨立的
    # preferred_caregiver_max_travel_minutes 上限——若該欄位亦為 None，代表歷史首選
    # 居服員完全不受車程上限約束，不會退回套用一般候選人的 max_travel_minutes，
    # 確保熟悉度高的居服員不因車程稍長就被硬性剔除。
    if travel_time_min is not None:
        cap = config.preferred_caregiver_max_travel_minutes if is_preferred_caregiver else config.max_travel_minutes
        if cap is not None and travel_time_min > cap:
            return f"預估車程({travel_time_min:.0f}分)超過上限({cap:.0f}分鐘)"

    required_certs = SPECIAL_CERT_REQUIREMENTS.get(task["特殊照護需求"])
    if required_certs and cg["核心專長證照"] not in required_certs:
        return "缺乏該需求類別之法定專長認證"

    return None


# ==========================================
# Phase 1: 適配度過濾與評分機制
# ==========================================
def run_phase1_matching(tasks: pd.DataFrame, df_cg: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    match_results = []

    # 進入逐筆比對迴圈前，先以單次 OSRM /table 批次查詢暖身快取（居服員住家 x 任務地點），
    # 取代 N x M 次個別 HTTP 請求。
    prefetch_osrm_travel_times(
        ((cg["服務起點_緯度(家)"], cg["服務起點_經度(家)"]) for _, cg in df_cg.iterrows()),
        ((task["服務地點_緯度"], task["服務地點_經度"]) for _, task in tasks.iterrows()),
        config.travel_min_per_km,
    )

    for _, task in tasks.iterrows():
        t_id = task["任務ID"]
        c_id = task["案家ID"]
        client_lat = task["服務地點_緯度"]
        client_lon = task["服務地點_經度"]
        pref_cg = task["歷史首選居服員ID"]
        req_type = task["特殊照護需求"]

        matched_count_for_task = 0
        reason_counts: dict = {}

            for _, cg in df_cg.iterrows():
                cg_id = cg["居服員ID"]
                is_preferred = cg_id == pref_cg

            transport_mode = str(
                cg.get("常用交通工具", "機車")
            ).strip()

            if transport_mode not in ("機車", "大眾運輸"):
                transport_mode = "機車"

            travel_time_min = calc_travel_minutes(
                cg["服務起點_緯度(家)"],
                cg["服務起點_經度(家)"],
                client_lat,
                client_lon,
                config,
                transport_mode=transport_mode,
            )

            # --- Hard Constraints (硬性過濾，不合格者直接剔除) ---
            err_msg = _check_hard_constraints(task, cg, config, travel_time_min, is_preferred)
            if err_msg is not None:
                reason_counts[err_msg] = reason_counts.get(err_msg, 0) + 1
                continue

            matched_count_for_task += 1
            cert = cg["核心專長證照"]

            # --- Soft Match Scoring ---
            # 0. 基礎分
            base_score = config.base_score

            # 1. 專長匹配
            skill_bonus = 0.0

            if req_type == "失智引導與精神陪伴" and cert in [
                "失智症照顧專長",
                "精神疾病照顧專長",
            ]:
                skill_bonus = config.cert_bonus_dementia

            elif (
                req_type in ["管路安全與特殊日常照護", "餐食照顧/管灌"]
                and cert == "單一級照服證照"
            ):
                skill_bonus = config.cert_bonus_other


            # 2. 照護連續性
            continuity_bonus = 0.0
            continuity_performance_bonus = 0.0

            if is_preferred:
                continuity_bonus = config.preferred_caregiver_bonus

                if (
                    cg["歷史滿意度均值"]
                    >= config.continuity_satisfaction_threshold
                ):
                    continuity_performance_bonus = (
                        config.continuity_performance_bonus
                    )


            # 3. 歷史服務品質
            satisfaction_adjustment = (
                cg["歷史滿意度均值"]
                - config.satisfaction_baseline
            ) * config.satisfaction_weight


            # 4. 交通成本
            travel_penalty = min(
                travel_time_min * config.travel_penalty_weight,
                config.travel_penalty_cap
            )


            # 5. 疲勞 / 工作負荷
            intensity_weight = get_service_intensity_weight(task)
            weighted_fatigue_hours = cg["當月累計服務時數(疲勞度)"] * intensity_weight
            fatigue_penalty = (
                 weighted_fatigue_hours / config.fatigue_reference_hours
            ) * config.fatigue_weight


            # ==========================================
            # 最終適配度分數
            # ==========================================
            score = (
                base_score
                + skill_bonus
                + continuity_bonus
                + continuity_performance_bonus
                + satisfaction_adjustment
                - travel_penalty
                - fatigue_penalty
            )


            # ==========================================
            # 保存結果
            # ==========================================
            match_results.append(
                {
                    "任務ID": t_id,
                    "案家ID": c_id,
                    "居服員ID": cg_id,

                    "適配度分數": round(max(score, 0), 2),

                    # Explainable AI：分數組成
                    "基礎分": round(base_score, 2),
                    "專長匹配加分": round(skill_bonus, 2),
                    "歷史首選加分": round(continuity_bonus, 2),
                    "連續性品質加分": round(
                        continuity_performance_bonus,
                        2
                    ),
                    "滿意度調整": round(
                        satisfaction_adjustment,
                        2
                    ),
                    "交通扣分": round(
                        travel_penalty,
                        2
                    ),
                    "工作負荷扣分": round(
                        fatigue_penalty,
                        2
                    ),

                    # Explainability 輔助資訊
                    "是否歷史首選": bool(is_preferred),

                    "當月累計服務時數": round(
                        float(
                            cg["當月累計服務時數(疲勞度)"]
                        ),
                        1
                    ),

                    "服務強度係數": round(
                        float(intensity_weight),
                        2
                    ),

                    # 原本欄位
                    "預估交通時間(分)": round(
                        travel_time_min,
                        1
                    ),

                    "任務開始時間":
                        task["時間窗_開始"],

                    "任務結束時間":
                        task["時間窗_結束"],

                    "優先級":
                        task["任務優先級"],

                    "地點緯度":
                        client_lat,

                    "地點經度":
                        client_lon,

                    "具備失智症20小時認證(0/1)":
                        int(
                            cert == "失智症照顧專長"
                        ),

                    "具備精神疾病20小時認證(0/1)":
                        int(
                            cert == "精神疾病照顧專長"
                        ),

                    "常用交通工具": transport_mode,
                }
            )
        if matched_count_for_task == 0:
            print(f"[Match Warning] 任務 {t_id} 找不到任何符合條件的居服員，剔除原因分佈：{reason_counts}")

    return pd.DataFrame(match_results)



def _diagnose_caregiver_change(
    task_row,
    pref_cg_id,
    assigned_cg_id,
    df_cg: pd.DataFrame,
    df_matches: pd.DataFrame,
    df_valid: pd.DataFrame,
    other_assigned_task_ids,
    task_times: dict,
    config: "PipelineConfig",
) -> str:
    """回傳「原首選替換原因」文字，說明*歷史首選居服員*為何未獲派本次任務。

    案家為新客戶（無歷史首選居服員）或本次仍指派給原首選居服員時回傳空字串；
    僅在確實更換居服員時才需要說明原因。

    回傳字串一律以 `原首選居服員[ID]` 開頭明確帶出主語，避免與「本次獲派居服員」
    混淆——過去訊息（如「居服員當日不在可排班星期名單內」）沒有主語，居督容易誤讀
    成是在描述獲派居服員不符資格，但其實描述的是原首選居服員被替換的原因。
    """
    if pd.isna(pref_cg_id) or str(pref_cg_id).strip() == "":
        return ""
    if pref_cg_id == assigned_cg_id:
        return ""

    t_id = task_row["任務ID"]
    subject = f"原首選居服員[{pref_cg_id}]"

    pref_in_matches = ((df_matches["任務ID"] == t_id) & (df_matches["居服員ID"] == pref_cg_id)).any()
    if not pref_in_matches:
        cg_rows = df_cg[df_cg["居服員ID"] == pref_cg_id]
        if cg_rows.empty:
            return f"{subject}資料異動，系統查無此居服員"
        cg_row = cg_rows.iloc[0]
        travel_time_min = calc_travel_minutes(
            cg_row["服務起點_緯度(家)"],
            cg_row["服務起點_經度(家)"],
            task_row["服務地點_緯度"],
            task_row["服務地點_經度"],
            config,
        )
        detail = (
            _check_hard_constraints(task_row, cg_row, config, travel_time_min, is_preferred_caregiver=True)
            or "不符合硬性派單條件"
        )
        return f"{subject}{detail}"

    pref_in_valid = ((df_valid["任務ID"] == t_id) & (df_valid["居服員ID"] == pref_cg_id)).any()
    if not pref_in_valid:
        return f"{subject}今日既定行程與本任務時段衝突"

    t_start, t_end, t_lat, t_lon = task_times[t_id]
    for other_t_id in other_assigned_task_ids:
        o_start, o_end, o_lat, o_lon = task_times[other_t_id]
        travel_mins = calc_travel_minutes(t_lat, t_lon, o_lat, o_lon, config) + config.buffer_mins
        if not (
            t_end + timedelta(minutes=travel_mins) <= o_start
            or o_end + timedelta(minutes=travel_mins) <= t_start
        ):
            return f"{subject}該時段已媒合其他案家任務"

    return f"{subject}雖符合派單資格，惟系統整體最佳化後綜合適配分數較低，已改派其他居服員"


def _evaluate_reassignment(
    task_id,
    new_cg_id,
    df_tasks: pd.DataFrame,
    df_result_effective: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: "PipelineConfig",
    task_locations: Optional[dict] = None,
    date_column: str = "日期",
    df_cl: Optional[pd.DataFrame] = None,
) -> dict:
    """評估把 task_id 改派給 new_cg_id 是否可行，回傳結構化結果供兩種呼叫端共用：

    - `check_reassignment_conflict`：改派當下的最終權威判定（僅需 available/detail）。
    - `rank_candidates_by_availability`：改派下拉選單的候選人清單排序與標籤
      （需要衝突任務的時段細節，才能顯示如「14:00-15:00 服務中」的具體標籤）。

    單一事實來源：兩種呼叫端看到的「是否可派」結論恆一致，不會有選單顯示可派、
    實際確認卻被拒絕的落差。

    回傳 dict 固定包含 "cg_id"、"available"、"reason"
    （None｜"task_not_found"｜"caregiver_not_found"｜"bad_time_format"｜
    "hard_constraint"｜"time_conflict"｜"hour_cap"）、"detail"（完整說明文字）；
    reason 為 "time_conflict" 時另附 "conflict_task_id"／"conflict_start"／"conflict_end"。

    `df_cl`（Client_Profiles）為選填：提供時會額外以 Phase 1 同一套
    `_check_hard_constraints`（性別限定、重度移位體力、環境排斥、可排班星期、
    每日可服務時段、請假日期、專長認證）評估 new_cg_id 是否符合本任務的硬性
    資格條件，失敗回傳 reason="hard_constraint" 並附上具體原因（例如「當日已有
    請假或不排班記錄」「缺乏該需求類別之法定專長認證」）——這只影響下拉選單的
    排序與標籤（見 rank_candidates_by_availability），刻意不接在
    check_reassignment_conflict 的權威判定路徑上：本系統沒有「強制派單權限」機制，
    居督仍可能因臨時狀況刻意指派不符合建議條件的居服員，故只標記不隱藏、不封鎖。
    不提供 df_cl（預設 None）時完全跳過此檢查，行為與加入前相同。
    """
    result = {
        "cg_id": new_cg_id,
        "available": False,
        "reason": None,
        "detail": "",
        "conflict_task_id": None,
        "conflict_start": None,
        "conflict_end": None,
    }

    task_rows = df_tasks[df_tasks["任務ID"] == task_id]
    if task_rows.empty:
        result.update(reason="task_not_found", detail="找不到該任務資料，無法檢查衝突")
        return result
    task_row = task_rows.iloc[0]

    cg_rows = df_cg[df_cg["居服員ID"].astype(str) == str(new_cg_id)]
    if cg_rows.empty:
        result.update(reason="caregiver_not_found", detail=f"找不到居服員 {new_cg_id} 的資料")
        return result
    cg_row = cg_rows.iloc[0]

    task_locations = task_locations or {}

    if df_cl is not None and not df_cl.empty and "案家ID" in task_row.index:
        cl_rows = df_cl[df_cl["案家ID"].astype(str) == str(task_row["案家ID"])]
        if not cl_rows.empty:
            merged_task = pd.Series({**cl_rows.iloc[0].to_dict(), **task_row.to_dict()})
            travel_time_min = None
            t_loc = task_locations.get(task_id)
            cg_home_lat = cg_row.get("服務起點_緯度(家)")
            cg_home_lon = cg_row.get("服務起點_經度(家)")
            if t_loc and pd.notna(cg_home_lat) and pd.notna(cg_home_lon) and all(pd.notna(v) for v in t_loc):
                travel_time_min = calc_travel_minutes(cg_home_lat, cg_home_lon, t_loc[0], t_loc[1], config)
            is_preferred = str(merged_task.get("歷史首選居服員ID", "")) == str(new_cg_id)
            hard_reason = _check_hard_constraints(merged_task, cg_row, config, travel_time_min, is_preferred)
            if hard_reason is not None:
                result.update(reason="hard_constraint", detail=hard_reason)
                return result

    t_start, t_end = parse_time(f"{task_row['時間窗_開始']}-{task_row['時間窗_結束']}")
    if t_start is None:
        result.update(reason="bad_time_format", detail="任務時間格式無法解析，無法檢查衝突")
        return result

    if date_column in df_tasks.columns and pd.notna(task_row.get(date_column)):
        task_date = task_row[date_column]
        same_day_ids = set(df_tasks.loc[df_tasks[date_column] == task_date, "任務ID"])
    else:
        same_day_ids = set(df_tasks["任務ID"])  # 單日排程：所有任務視為同一天

    other_assigned = (
        df_result_effective[
            (df_result_effective["派單居服員"].astype(str) == str(new_cg_id))
            & (df_result_effective["任務ID"].isin(same_day_ids))
            & (df_result_effective["任務ID"] != task_id)
        ]
        if not df_result_effective.empty
        else df_result_effective
    )

    total_minutes = float(task_row["服務歷時(分鐘)"])

    if other_assigned is not None:
        for other_t_id in other_assigned["任務ID"]:
            other_rows = df_tasks[df_tasks["任務ID"] == other_t_id]
            if other_rows.empty:
                continue
            other_row = other_rows.iloc[0]
            o_start, o_end = parse_time(f"{other_row['時間窗_開始']}-{other_row['時間窗_結束']}")
            if o_start is None:
                continue

            travel_mins = config.buffer_mins
            t_loc = task_locations.get(task_id)
            o_loc = task_locations.get(other_t_id)
            if t_loc and o_loc and all(pd.notna(v) for v in (*t_loc, *o_loc)):
                travel_mins += calc_travel_minutes(t_loc[0], t_loc[1], o_loc[0], o_loc[1], config)

            if not (
                t_end + timedelta(minutes=travel_mins) <= o_start
                or o_end + timedelta(minutes=travel_mins) <= t_start
            ):
                result.update(
                    reason="time_conflict",
                    conflict_task_id=other_t_id,
                    conflict_start=str(other_row["時間窗_開始"]),
                    conflict_end=str(other_row["時間窗_結束"]),
                    detail=(
                        f"與居服員 {new_cg_id} 當日另一任務（{other_t_id}，"
                        f"{other_row['時間窗_開始']}-{other_row['時間窗_結束']}）時間衝突"
                        f"（含轉場緩衝約 {travel_mins:.0f} 分鐘）"
                    ),
                )
                return result
            total_minutes += float(other_row["服務歷時(分鐘)"])

    daily_cap = cg_row.get("每日工時上限(小時)")
    if pd.notna(daily_cap) and total_minutes / 60.0 > daily_cap:
        result.update(
            reason="hour_cap",
            detail=(
                f"居服員 {new_cg_id} 改派後當日總工時將達 {total_minutes / 60.0:.1f} 小時，"
                f"超過每日上限 {daily_cap:.1f} 小時"
            ),
        )
        return result

    result["available"] = True
    return result


def check_reassignment_conflict(
    task_id,
    new_cg_id,
    df_tasks: pd.DataFrame,
    df_result_effective: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: "PipelineConfig",
    task_locations: Optional[dict] = None,
    date_column: str = "日期",
) -> Optional[str]:
    """檢查「快速改派」把 task_id 轉給 new_cg_id 是否會造成時間衝突或工時超標。

    供月曆視角一鍵調班（calendar_view.py）使用：`df_result_effective` 須為已套用
    居督覆寫後的『目前生效』派單結果（見 apply_overrides_to_result），確保衝突
    檢查基準與畫面顯示一致，不會用「AI 原始建議」誤判已被覆寫過的任務。

    衝突判定與 Phase 2 限制條件 2（同一居服員新任務時間不得重疊，須預留
    config.buffer_mins 轉場緩衝）採同一公式，僅多檢查每日工時上限。
    `task_locations`（任務ID -> (緯度, 經度)，通常取自 df_matches）用於估算轉場
    車程；缺少座標時保守僅以 config.buffer_mins 判斷重疊，不會略過檢查。

    回傳 None 表示可安全改派；否則回傳供 UI 顯示的錯誤說明文字。
    """
    result = _evaluate_reassignment(
        task_id, new_cg_id, df_tasks, df_result_effective, df_cg, config, task_locations, date_column
    )
    return None if result["available"] else result["detail"]


def rank_candidates_by_availability(
    task_id,
    candidate_cg_ids,
    df_tasks: pd.DataFrame,
    df_result_effective: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: "PipelineConfig",
    task_locations: Optional[dict] = None,
    date_column: str = "日期",
    df_cl: Optional[pd.DataFrame] = None,
) -> list:
    """供月曆快速改派下拉選單使用：把候選居服員依「該時段是否有空檔」排序。

    對 candidate_cg_ids 逐一呼叫 `_evaluate_reassignment`（與確認改派時
    check_reassignment_conflict 走同一套判定，避免選單顯示可派、實際確認卻被
    拒絕的落差），回傳依 available 由高到低排序（同組內維持 candidate_cg_ids
    原始順序）的 dict list，每筆結構同 `_evaluate_reassignment` 的回傳值。

    candidate_cg_ids 預期為「機構內全體居服員」（呼叫端不應預先用 Phase 1
    的硬性條件篩過一輪才傳進來，否則不符合資格的人會直接從清單消失，而非
    保留＋標記）；提供 `df_cl` 時，本函式會連同 Phase 1 的硬性資格條件
    （性別限定、重度移位體力、環境排斥、可排班星期、可服務時段、請假日期、
    專長認證）一併標記為 reason="hard_constraint"，而非讓呼叫端事先濾掉。
    刻意不將此檢查接上 check_reassignment_conflict（該函式呼叫
    _evaluate_reassignment 時不傳 df_cl）：本系統無強制派單權限機制，居督仍可
    在清楚看到標記後選擇覆寫，故硬性資格條件在此僅供標籤顯示，不封鎖改派。
    """
    evaluated = [
        _evaluate_reassignment(
            task_id, cg_id, df_tasks, df_result_effective, df_cg, config,
            task_locations, date_column, df_cl,
        )
        for cg_id in candidate_cg_ids
    ]
    return sorted(evaluated, key=lambda r: not r["available"])


def _build_cg_busy_blocks(df_cg: pd.DataFrame) -> dict:
    """建立居服員今日既定行程時間阻擋塊 (Time Blocks)，回傳 cg_id -> [(start, end, lat, lon), ...]。

    「今日既定行程」欄位為新版月批次排班資料表所無（改以可排班星期/請假日期取代），
    以 `.get()` 取值使兩種資料表格式皆可安全運作：欄位不存在時回傳 None，
    parse_time 會將 None 視為「無既定行程」而正確跳過。
    """
    cg_busy = {}
    for _, cg in df_cg.iterrows():
        cg_id = cg["居服員ID"]
        busy_intervals = []
        for i in [1, 2]:
            t1, t2 = parse_time(cg.get(f"今日既定行程{i}_時段"))
            if t1:
                busy_intervals.append(
                    (
                        t1,
                        t2,
                        cg.get(f"今日既定行程{i}_地點緯度"),
                        cg.get(f"今日既定行程{i}_地點經度"),
                    )
                )
        cg_busy[cg_id] = busy_intervals
    return cg_busy


def _build_break_constraints(solver, X: dict, df_valid: pd.DataFrame, cg_busy: dict, task_times: dict, config: "PipelineConfig") -> None:
    """規則1（勞動合規）：居服員累計連續工作達 config.continuous_work_limit_mins 分鐘，
    強制要求下一段任務與前段之間至少間隔 config.mandatory_break_mins 分鐘。

    做法：對每位居服員，把「今日既定行程」（固定發生）與「通過衝突檢查的候選新任務」
    （指派變數）依開始時間排序，切出「潛在連續鏈」——鏈內相鄰兩區塊的固定時間差
    < mandatory_break_mins。對鏈中每一個起點，往後累加工作分鐘數直到達到門檻，
    即對緊接在後的區塊加入限制式，禁止「該起點到達門檻的前綴」與「緊接的下一個
    候選任務」同時獲派（前綴若全為既有既定行程，則後續候選任務直接被禁止指派）。

    此為保守近似：以「潛在鏈上的固定時間」計算，而非僅以「實際獲派子集合」重新
    計算，在極少數「跳過鏈中某段候選任務即可讓實際連續工時縮短」的邊界情況下可能
    偏嚴（阻擋一個其實合規的組合），但休息規則屬勞動合規要求，寧可偏保守也不可
    漏判真違規；居督仍可透過既有人工覆寫機制（save_override_log）調整結果。
    """
    for cg_id in df_valid["居服員ID"].unique():
        blocks = []
        for b_start, b_end, _b_lat, _b_lon in cg_busy.get(cg_id, []):
            blocks.append(
                {"start": b_start, "end": b_end, "duration_min": (b_end - b_start).total_seconds() / 60.0, "var": None}
            )
        for t_id in df_valid[df_valid["居服員ID"] == cg_id]["任務ID"].unique():
            t_start, t_end, _, _ = task_times[t_id]
            blocks.append(
                {
                    "start": t_start,
                    "end": t_end,
                    "duration_min": (t_end - t_start).total_seconds() / 60.0,
                    "var": X[(t_id, cg_id)],
                }
            )
        blocks.sort(key=lambda b: b["start"])

        # 依固定時間切出潛在連續鏈：鏈內相鄰區塊時間差 < mandatory_break_mins
        i = 0
        n = len(blocks)
        while i < n:
            j = i + 1
            while j < n:
                gap_mins = (blocks[j]["start"] - blocks[j - 1]["end"]).total_seconds() / 60.0
                if gap_mins >= config.mandatory_break_mins:
                    break
                j += 1
            chain = blocks[i:j]

            # 對鏈中每個起點 k，找出往後累加達門檻的最短前綴 [k..m]，並限制其後一個
            # 候選任務不得與該前綴同時獲派。
            for k in range(len(chain)):
                cum = 0.0
                m = None
                for idx in range(k, len(chain)):
                    cum += chain[idx]["duration_min"]
                    if cum >= config.continuous_work_limit_mins:
                        m = idx
                        break
                if m is None or m + 1 >= len(chain):
                    continue

                extra = chain[m + 1]
                if extra["var"] is None:
                    continue  # 既有既定行程本身即固定發生，無法以指派變數禁止

                prefix_vars = [b["var"] for b in chain[k : m + 1] if b["var"] is not None]
                if not prefix_vars:
                    solver.Add(extra["var"] == 0)
                else:
                    solver.Add(sum(prefix_vars) + extra["var"] <= len(prefix_vars))

            i = j


# ==========================================
# Phase 2: 時空路徑衝突過濾 + OR-Tools 多目標最佳化
# ==========================================
def run_phase2_optimization(
    df_matches: pd.DataFrame,
    tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: PipelineConfig,
    extra_busy_blocks: Optional[dict] = None,
):
    """執行 Phase 2 時空衝突過濾與 OR-Tools 最佳化派單。

    extra_busy_blocks（cg_id -> [(start, end, lat, lon), ...]，可選）用於「週期性任務
    優先」排班：由呼叫端（見 `_run_periodic_then_adhoc`）將前一輪已鎖定的週期性任務
    指派結果，轉為額外的忙碌時間區塊注入本輪臨時單次任務的衝突檢查，使臨時任務
    只能競爭週期性任務排定後剩餘的時段，不會與其重疊。
    """
    # 建立居服員今日既定行程時間阻擋塊 (Time Blocks)。
    cg_busy = _build_cg_busy_blocks(df_cg)
    if extra_busy_blocks:
        for cg_id, blocks in extra_busy_blocks.items():
            cg_busy.setdefault(cg_id, []).extend(blocks)

    # 建立任務時間表
    task_times = {}
    for _, task in tasks.iterrows():
        t_id = task["任務ID"]
        t1 = datetime.strptime(task["時間窗_開始"].strip(), "%H:%M")
        t2 = datetime.strptime(task["時間窗_結束"].strip(), "%H:%M")
        task_times[t_id] = (t1, t2, task["服務地點_緯度"], task["服務地點_經度"])

    # 進入衝突檢查迴圈前，先以 OSRM /table 批次查詢暖身快取（任務地點 x 既定行程地點、
    # 任務地點 x 任務地點），取代逐筆個別 HTTP 請求。
    task_coords = [(lat, lon) for _, _, lat, lon in task_times.values()]
    busy_coords = [
        (b_lat, b_lon) for intervals in cg_busy.values() for (_, _, b_lat, b_lon) in intervals
    ]
    if busy_coords:
        prefetch_osrm_travel_times(task_coords, busy_coords, config.travel_min_per_km)
    prefetch_osrm_travel_times(task_coords, task_coords, config.travel_min_per_km)

    # 過濾掉與既定行程衝突（含轉場緩衝時間）的配對
    valid_rows = []
    for _, row in df_matches.iterrows():
        t_id = row["任務ID"]
        cg_id = row["居服員ID"]
        t_start, t_end, t_lat, t_lon = task_times[t_id]

        conflict = False
        for b_start, b_end, b_lat, b_lon in cg_busy[cg_id]:
            travel_mins = calc_travel_minutes(t_lat, t_lon, b_lat, b_lon, config) + config.buffer_mins
            if not (
                t_end + timedelta(minutes=travel_mins) <= b_start
                or b_end + timedelta(minutes=travel_mins) <= t_start
            ):
                conflict = True
                break

        if not conflict:
            valid_rows.append(row.to_dict())

    df_valid = pd.DataFrame(valid_rows)

    result = {
        "df_valid": df_valid,
        "status": None,
        "df_result": pd.DataFrame(),
        "assigned_count": 0,
    }

    if df_valid.empty:
        return result

    # 建立 OR-Tools 混合整數規劃 (MIP) 求解器
    solver = pywraplp.Solver.CreateSolver("SCIP")

    X = {}
    for _, row in df_valid.iterrows():
        X[(row["任務ID"], row["居服員ID"])] = solver.IntVar(
            0, 1, f"x_{row['任務ID']}_{row['居服員ID']}"
        )

    # 限制條件 1：每個任務至多只能派給一位居服員
    for t_id in df_valid["任務ID"].unique():
        solver.Add(
            sum(
                X[(t_id, cg_id)]
                for cg_id in df_valid[df_valid["任務ID"] == t_id]["居服員ID"]
            )
            <= 1
        )

    # 限制條件 2：同一居服員若被派兩個新任務，時間不能重疊且須預留轉場緩衝
    for cg_id in df_valid["居服員ID"].unique():
        cg_tasks = df_valid[df_valid["居服員ID"] == cg_id]["任務ID"].tolist()
        for i in range(len(cg_tasks)):
            for j in range(i + 1, len(cg_tasks)):
                t1_id, t2_id = cg_tasks[i], cg_tasks[j]
                t1_start, t1_end, t1_lat, t1_lon = task_times[t1_id]
                t2_start, t2_end, t2_lat, t2_lon = task_times[t2_id]

                travel_mins = calc_travel_minutes(t1_lat, t1_lon, t2_lat, t2_lon, config) + config.buffer_mins

                if not (
                    t1_end + timedelta(minutes=travel_mins) <= t2_start
                    or t2_end + timedelta(minutes=travel_mins) <= t1_start
                ):
                    solver.Add(X[(t1_id, cg_id)] + X[(t2_id, cg_id)] <= 1)

    # 限制條件 2b：規則1（4小時/30分鐘休息）——累計連續工作達門檻須強制安插休息。
    _build_break_constraints(solver, X, df_valid, cg_busy, task_times, config)

    # 限制條件 3：居服員今日新派任務總歷時 + 既有已佔用工時，不得超過每日工時上限。
    # Phase 1 僅逐筆過濾單一任務是否超時，無法阻擋「多筆任務加總後超派」的組合，
    # 此為求解器層級的產能限制式，修復該缺口。
    task_duration_hrs = {
        row["任務ID"]: row["服務歷時(分鐘)"] / 60.0 for _, row in tasks.iterrows()
    }
    cg_capacity = {
        row["居服員ID"]: (row.get("今日已佔用工時(小時)", 0.0), row["每日工時上限(小時)"])
        for _, row in df_cg.iterrows()
    }
    for cg_id in df_valid["居服員ID"].unique():
        used_hours, cap_hours = cg_capacity.get(cg_id, (0.0, float("inf")))
        cg_task_ids = df_valid[df_valid["居服員ID"] == cg_id]["任務ID"].unique()
        solver.Add(
            sum(X[(t_id, cg_id)] * task_duration_hrs[t_id] for t_id in cg_task_ids)
            + used_hours
            <= cap_hours
        )

    # 財務試算：每筆任務的長照申報點數（營收）與居服員拆帳薪資，
    # 僅供結果顯示與財務 KPI 試算，不參與派單目標函數。
    tasks_with_revenue = calculate_task_revenue_and_salary(tasks, config)
    task_revenue_map = {
        row["任務ID"]: (row["預估長照申報點數(營收)"], row["預估居服員拆帳薪資"])
        for _, row in tasks_with_revenue.iterrows()
    }

    # Phase 2 目標函數：
    # 在通過所有硬性條件與時空衝突檢查的候選方案中，
    # 最大化 Phase 1 適配度分數與任務優先權。
    # 車程已納入 Phase 1 適配度計算，不重複扣分；
    # 財務營收僅供結果顯示與 KPI 試算，不參與派單決策。
    #
    # 「照護連續性」優先於「車程限制」的原則亦須貫徹到此目標函數層級：若僅單純調高
    # 反而會讓車程較遠但為案家歷史首選的居服員，在整體最佳化階段被距離較近的陌生
    # 居服員取代——這違背了 Phase 1 刻意給予首選居服員高額連續性加分的用意（其車程
    # 成本已由 Phase 1 的 travel_penalty_cap 合理封頂）。因此歷史首選居服員的配對在
    objective = solver.Objective()
    for _, row in df_valid.iterrows():
        t_id = row["任務ID"]
        cg_id = row["居服員ID"]
        w_match = row["適配度分數"]
        priority_bonus = (
            config.urgent_priority_bonus
            if "高" in str(row["優先級"])
            else config.normal_priority_bonus
        )

        coeff = w_match  + priority_bonus 
        objective.SetCoefficient(X[(t_id, cg_id)], coeff)

    objective.SetMaximization()

    status = solver.Solve()
    result["status"] = status

    if status == pywraplp.Solver.OPTIMAL:
        assigned_count = 0
        results = []
        for (t_id, cg_id), var in X.items():
            if var.solution_value() > 0.5:
                assigned_count += 1
                row_data = df_valid[
                    (df_valid["任務ID"] == t_id) & (df_valid["居服員ID"] == cg_id)
                ].iloc[0]
                revenue, salary = task_revenue_map.get(t_id, (0.0, 0.0))
                results.append(
                    {
                        "任務ID": t_id,
                        "案家ID": row_data["案家ID"],
                        "派單居服員": cg_id,
                        "適配分數": row_data["適配度分數"],
                        "預估車程(分)": row_data["預估交通時間(分)"],
                        "服務時段": f"{row_data['任務開始時間']}-{row_data['任務結束時間']}",
                        "任務優先級": row_data["優先級"],
                        "地點緯度": row_data["地點緯度"],
                        "地點經度": row_data["地點經度"],
                        "預估長照申報點數(營收)": revenue,
                        "預估居服員拆帳薪資": salary,
                        "基礎分": row_data.get("基礎分", 0),

                        "專長匹配加分": row_data.get("專長匹配加分", 0),
                        "歷史首選加分": row_data.get("歷史首選加分", 0),
                        "連續性品質加分": row_data.get("連續性品質加分", 0),
                        "滿意度調整": row_data.get("滿意度調整", 0),
                        "交通扣分": row_data.get("交通扣分", 0),
                        "工作負荷扣分": row_data.get("工作負荷扣分", 0),
                        "是否歷史首選": row_data.get("是否歷史首選", False),
                        "當月累計服務時數": row_data.get("當月累計服務時數", 0),
                        "服務強度係數": row_data.get("服務強度係數", 1),
                    }
                )

        # 每位居服員本次新指派到的任務清單，供「原首選替換原因」判斷同時段衝突用
        assigned_task_ids_by_cg = {}
        for row in results:
            assigned_task_ids_by_cg.setdefault(row["派單居服員"], []).append(row["任務ID"])

        for row in results:
            t_id = row["任務ID"]
            task_row = tasks[tasks["任務ID"] == t_id].iloc[0]
            pref_cg_id = task_row["歷史首選居服員ID"]
            other_task_ids = [
                tid for tid in assigned_task_ids_by_cg.get(pref_cg_id, []) if tid != t_id
            ]
            row["原首選替換原因"] = _diagnose_caregiver_change(
                task_row,
                pref_cg_id,
                row["派單居服員"],
                df_cg,
                df_matches,
                df_valid,
                other_task_ids,
                task_times,
                config,
            )

        df_result = pd.DataFrame(results)
        if not df_result.empty:
            df_result = df_result.sort_values(by="任務ID")
        result["df_result"] = df_result
        result["assigned_count"] = assigned_count

    return result


PERIODIC_TASK_COLUMN = "是否為週期性任務"


def _is_periodic_task(value) -> bool:
    """將「是否為週期性任務」欄位值正規化為布林值。

    Excel 布林儲存格經 pandas 讀入後，實際型別可能是原生 Python bool（勾選格式）、
    字串 "TRUE"/"FALSE"（文字格式儲存格）、或 1/0；本函式統一正規化為 bool，
    空白（NaN）保守視為非週期性（臨時單次）任務。
    """
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().upper() == "TRUE"
    return bool(value)


def _split_periodic_tasks(tasks: pd.DataFrame, periodic_column: str = PERIODIC_TASK_COLUMN):
    """依 periodic_column 將 tasks 拆分為 (週期性任務, 臨時單次任務)。

    欄位不存在時回傳 (None, None)，代表呼叫端應維持原本單一批次排班邏輯
    （向下相容不含此欄位的舊版資料表）。
    """
    if periodic_column not in tasks.columns:
        return None, None
    is_periodic = tasks[periodic_column].apply(_is_periodic_task)
    return tasks[is_periodic].copy(), tasks[~is_periodic].copy()


def _run_periodic_then_adhoc(
    periodic_tasks: pd.DataFrame,
    adhoc_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: PipelineConfig,
) -> dict:
    """規則2：週期性排班優先於臨時單次排班。

    先只用週期性任務跑一輪 Phase 1 + Phase 2，鎖定基礎班表；再將該輪指派結果轉為
    額外忙碌時間區塊與已佔用工時，注入臨時單次任務的第二輪 Phase 1 + Phase 2，
    使臨時任務只能競爭週期性任務排定後「剩餘的居服員產能與時段」，不會與其重疊
    或反過來排擠週期性任務。
    """
    empty_result = {"df_valid": pd.DataFrame(), "status": None, "df_result": pd.DataFrame(), "assigned_count": 0}

    periodic_result = empty_result
    if not periodic_tasks.empty:
        periodic_matches = run_phase1_matching(periodic_tasks, df_cg, config)
        periodic_result = run_phase2_optimization(periodic_matches, periodic_tasks, df_cg, config)

    df_cg_adhoc = df_cg.copy()
    extra_busy_blocks: dict = {}
    df_periodic_result = periodic_result["df_result"]
    if not df_periodic_result.empty:
        if "今日已佔用工時(小時)" not in df_cg_adhoc.columns:
            df_cg_adhoc["今日已佔用工時(小時)"] = 0.0

        periodic_tasks_by_id = periodic_tasks.set_index("任務ID")
        hours_by_cg: dict = {}
        for _, row in df_periodic_result.iterrows():
            t_id = row["任務ID"]
            cg_id = row["派單居服員"]
            task_row = periodic_tasks_by_id.loc[t_id]
            t_start = datetime.strptime(str(task_row["時間窗_開始"]).strip(), "%H:%M")
            t_end = datetime.strptime(str(task_row["時間窗_結束"]).strip(), "%H:%M")
            extra_busy_blocks.setdefault(cg_id, []).append(
                (t_start, t_end, task_row["服務地點_緯度"], task_row["服務地點_經度"])
            )
            hours_by_cg[cg_id] = hours_by_cg.get(cg_id, 0.0) + task_row["服務歷時(分鐘)"] / 60.0

        for cg_id, hrs in hours_by_cg.items():
            mask = df_cg_adhoc["居服員ID"] == cg_id
            df_cg_adhoc.loc[mask, "今日已佔用工時(小時)"] = (
                df_cg_adhoc.loc[mask, "今日已佔用工時(小時)"].fillna(0.0) + hrs
            )

    adhoc_result = empty_result
    if not adhoc_tasks.empty:
        adhoc_matches = run_phase1_matching(adhoc_tasks, df_cg_adhoc, config)
        adhoc_result = run_phase2_optimization(
            adhoc_matches, adhoc_tasks, df_cg_adhoc, config, extra_busy_blocks=extra_busy_blocks
        )

    df_valid = pd.concat([periodic_result["df_valid"], adhoc_result["df_valid"]], ignore_index=True)
    df_result = pd.concat([df_periodic_result, adhoc_result["df_result"]], ignore_index=True)
    if not df_result.empty:
        df_result = df_result.sort_values(by="任務ID")

    return {
        "df_valid": df_valid,
        "df_result": df_result,
        "status": adhoc_result["status"] if not adhoc_tasks.empty else periodic_result["status"],
        "assigned_count": periodic_result["assigned_count"] + adhoc_result["assigned_count"],
    }


# ==========================================
# 全月/按日批次派單：對每個日期各自執行 Phase 1 + Phase 2
# ==========================================
def run_monthly_batch_dispatch(
    tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: PipelineConfig,
    date_column: str = "日期",
) -> dict:
    """依 `date_column`（預設「日期」）逐日切分 tasks，對每個日期各自獨立執行
    Phase 1 適配度評分與 Phase 2 OR-Tools 最佳化，再彙整為全期間派單總表。

    採「逐日各自求解」而非把整月任務一次丟進單一 MIP，原因有二：(1) 居服員的
    每日工時上限、既定行程衝突等限制式本質上就是以「日」為單位獨立成立，不同
    日期之間彼此不影響，逐日求解不損失最適性；(2) 任務規模隨天數線性成長時，
    單一大型 MIP 的求解時間會遠超逐日拆解後個別求解時間的總和。

    僅適用於 tasks 具備 date_column 欄位的月批次排班資料表（例如
    「Monthly_Pending_Tasks」）；若傳入不含該欄位的單日排班資料表，會直接拋出
    KeyError，提醒呼叫端改用 run_phase1_matching / run_phase2_optimization
    的單次呼叫方式。

    規則2（週期性排班優先）：若 tasks 具備 PERIODIC_TASK_COLUMN
    （"是否為週期性任務"）欄位，每天會先鎖定週期性任務的班表，臨時單次任務只能
    競爭剩餘產能／時段（見 `_run_periodic_then_adhoc`）；欄位不存在則維持原本
    單一批次邏輯，向下相容不含此欄位的資料表。

    規則5（月排班動態時數滾動）：內部維護一份 df_cg 的可變工作副本
    `df_cg_working`，每完成一天的指派後，將當天各居服員實際獲派的服務時數
    累加回其「當月累計服務時數(疲勞度)」欄位，供隔天 Phase 1 疲勞度懲罰納入計算
    （偏好派給累計時數較少者），以達到勞逸均衡；傳入的 df_cg 本身不會被修改。
    """
    daily_results: dict = {}
    result_frames = []
    total_assigned_count = 0

    df_cg_working = df_cg.copy()
    if "當月累計服務時數(疲勞度)" not in df_cg_working.columns:
        df_cg_working["當月累計服務時數(疲勞度)"] = 0.0

    unique_dates = sorted(tasks[date_column].dropna().unique())
    for target_date in unique_dates:
        day_tasks = tasks[tasks[date_column] == target_date].copy()
        if day_tasks.empty:
            continue

        periodic_tasks, adhoc_tasks = _split_periodic_tasks(day_tasks)
        if periodic_tasks is None:
            df_matches_daily = run_phase1_matching(day_tasks, df_cg_working, config)
            phase2_daily = run_phase2_optimization(df_matches_daily, day_tasks, df_cg_working, config)
        else:
            phase2_daily = _run_periodic_then_adhoc(periodic_tasks, adhoc_tasks, df_cg_working, config)

        daily_results[target_date] = phase2_daily

        df_daily_result = phase2_daily["df_result"]
        if not df_daily_result.empty:
            df_daily_result = df_daily_result.copy()
            df_daily_result["排班日期"] = target_date
            result_frames.append(df_daily_result)
            total_assigned_count += phase2_daily["assigned_count"]

            day_tasks_by_id = day_tasks.set_index("任務ID")
            hours_by_cg: dict = {}
            for _, row in df_daily_result.iterrows():
                t_id = row["任務ID"]
                cg_id = row["派單居服員"]
                duration_hrs = day_tasks_by_id.loc[t_id, "服務歷時(分鐘)"] / 60.0
                hours_by_cg[cg_id] = hours_by_cg.get(cg_id, 0.0) + duration_hrs
            for cg_id, hrs in hours_by_cg.items():
                mask = df_cg_working["居服員ID"] == cg_id
                df_cg_working.loc[mask, "當月累計服務時數(疲勞度)"] += hrs

    df_result_all = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()

    return {
        "daily_results": daily_results,
        "df_result_all": df_result_all,
        "total_assigned_count": total_assigned_count,
        "total_task_count": len(tasks),
    }


# ==========================================
# Phase 3: 效益產出評估 (DiD 雙重差分比較)
# ==========================================
def run_phase3_did(df_hist: pd.DataFrame) -> dict:
    ai_group = df_hist[df_hist["歷史媒合機制(Treatment)"] == 1]
    human_group = df_hist[df_hist["歷史媒合機制(Treatment)"] == 0]

    ai_sat = ai_group["案家滿意度(1-5)"].mean()
    human_sat = human_group["案家滿意度(1-5)"].mean()
    ai_dropout = ai_group["不滿意導致提早結案(0/1)"].mean()
    human_dropout = human_group["不滿意導致提早結案(0/1)"].mean()

    return {
        "ai_group": ai_group,
        "human_group": human_group,
        "ai_sat": ai_sat,
        "human_sat": human_sat,
        "ai_dropout": ai_dropout,
        "human_dropout": human_dropout,
        "uplift_sat": ai_sat - human_sat,
        "uplift_dropout": human_dropout - ai_dropout,
    }


# ==========================================
# 居督人工覆寫稽核日誌 (Human-in-the-Loop override audit log)
# ==========================================
OVERRIDE_LOG_PATH = os.path.join("output_results", "supervisor_override_log.csv")

OVERRIDE_LOG_COLUMNS = [
    "時間戳記", "任務ID", "案家ID", "AI推薦居服員ID", "居督指定居服員ID", "變更原因",
]


def save_override_log(
    task_id,
    client_id,
    ai_cg_id,
    supervisor_cg_id,
    reason: str,
    log_path: str = OVERRIDE_LOG_PATH,
) -> None:
    """將居督一筆人工覆寫紀錄以附加 (append) 方式寫入 CSV 稽核日誌。

    每次呼叫寫入一列；檔案不存在時先建立目錄與標題列。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = pd.DataFrame(
        [{
            "時間戳記": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "任務ID": task_id,
            "案家ID": client_id,
            "AI推薦居服員ID": ai_cg_id,
            "居督指定居服員ID": supervisor_cg_id,
            "變更原因": reason,
        }],
        columns=OVERRIDE_LOG_COLUMNS,
    )
    write_header = not os.path.exists(log_path)
    entry.to_csv(log_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def load_override_log(log_path: str = OVERRIDE_LOG_PATH) -> pd.DataFrame:
    """讀取現有稽核日誌；檔案不存在時回傳空的標準欄位 DataFrame。"""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=OVERRIDE_LOG_COLUMNS)
    return pd.read_csv(log_path, encoding="utf-8-sig")
