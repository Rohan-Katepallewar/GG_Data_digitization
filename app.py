import io, os, re, json, base64, time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# CV
import cv2

# Optional Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except Exception:
    HAS_SHEETS = False

# ====================== CONFIG / UI ======================
st.set_page_config(page_title="Kannada Paper Tool Digitizer (HIL)", page_icon="🧾", layout="wide")
st.title("🧾 Kannada Paper Tool → JSON / CSV / Google Sheets (Human-in-the-loop)")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    st.divider()
    st.subheader("Google Sheets (optional)")
    spreadsheet_id_input = st.text_input("Spreadsheet ID or URL", value="1rVMPiaZ5eeATiPTUJc4sQHEJMYZ8BPvsweEn7NSoFZM")
    worksheet_name = st.text_input("Worksheet/Tab (rows)", value="sheet1")

    st.divider()
    st.subheader("Detection controls")
    mark_thresh = st.slider("Mark detection threshold (ink ratio)", 0.05, 0.30, 0.12, 0.01)
    show_overlay = st.checkbox("Show debug overlay of regions", value=False)
    extra_digit_pass = st.checkbox("Extra digit pass for SATS/phones/UDISE", value=False,
                                   help="Crops number fields and asks the model for digits-only")

uploaded = st.file_uploader("Upload JPEG files (max 10)", type=["jpg","jpeg"], accept_multiple_files=True)

# ====================== CONSTANTS ======================
ALLOWED_OPS = ["None","Addition","Subtraction","Multiplication","Division"]
EDIT_OPS = ["", "Addition","Subtraction","Multiplication","Division"]  # for editing dropdowns
EDIT_YN  = ["", "Yes","No"]

ROW_COLUMNS = [
    "fileName","teacherName","schoolName","udise","cycle",
    "studentName","sats","caregiverName","studentPhone","caregiverPhone","grade",
    "baselineOp","session","datetime","currentTopic","checkpointCorrect","parentAttended","nextTopic","endlineOp"
]

# Student strip vertical ranges as FRACTIONS of page height (tune once if needed)
STUDENT_STRIP_FRACS = [
    (0.095, 0.275),   # Student 1
    (0.285, 0.465),   # Student 2
    (0.475, 0.655),   # Student 3
    (0.665, 0.845),   # Student 4
]

# Within each student strip, fractional boxes for the 5 ops (baseline row band)
# and 5 ops (endline row band). These are relative to the student strip (0..1).
BASELINE_OP_BOXES = [
    ("None",           0.20, 0.13, 0.27, 0.18),
    ("Addition",       0.30, 0.13, 0.36, 0.18),
    ("Subtraction",    0.39, 0.13, 0.47, 0.18),
    ("Multiplication", 0.50, 0.13, 0.60, 0.18),
    ("Division",       0.63, 0.13, 0.71, 0.18),
]
ENDLINE_OP_BOXES = [
    ("None",           0.20, 0.83, 0.27, 0.88),
    ("Addition",       0.30, 0.83, 0.36, 0.88),
    ("Subtraction",    0.39, 0.83, 0.47, 0.88),
    ("Multiplication", 0.50, 0.83, 0.60, 0.88),
    ("Division",       0.63, 0.83, 0.71, 0.88),
]

# Header/field crops for the extra digit pass (fractions of full page)
HEADER_UDISE_BOX = (0.75, 0.03, 0.96, 0.08)

# Per-student tiny crops within strip (fractions of strip)
SAT_BOX         = (0.12, 0.06, 0.28, 0.12)
STUDENT_PHONE   = (0.33, 0.06, 0.53, 0.12)
CAREGIVER_PHONE = (0.58, 0.06, 0.78, 0.12)

# ====================== PROMPTS ======================
EXTRACTION_PROMPT = """
You are an OCR + information extraction agent for Kannada handwriting. The form layout is fixed (4 student blocks).

Return ONLY a single JSON object with these keys exactly:
{
  "teacherName": "", "schoolName": "", "udise": "", "cycle": "",
  "students": [
    {
      "name": "", "sats": "", "caregiverName": "", "studentPhone": "", "caregiverPhone": "", "grade": "",
      "baselineMarks": {"None":false,"Addition":false,"Subtraction":false,"Multiplication":false,"Division":false},
      "endlineMarks":  {"None":false,"Addition":false,"Subtraction":false,"Multiplication":false,"Division":false},
      "baselineOp": "",
      "sessions": [
        {"session":"Lesson 1","datetime":"","currentTopic":"","checkpointCorrect":"","parentAttended":"","nextTopic":""},
        {"session":"Lesson 2","datetime":"","currentTopic":"","checkpointCorrect":"","parentAttended":"","nextTopic":""},
        {"session":"Lesson 3","datetime":"","currentTopic":"","checkpointCorrect":"","parentAttended":"","nextTopic":""},
        {"session":"Lesson 4","datetime":"","currentTopic":"","checkpointCorrect":"","parentAttended":"","nextTopic":""}
      ],
      "endlineOp": ""
    }
  ],
  "footer": {
    "lesson1":{"totalStudents":"","successfullyReached":""},
    "lesson2":{"totalStudents":"","successfullyReached":""},
    "lesson3":{"totalStudents":"","successfullyReached":""},
    "lesson4":{"totalStudents":"","successfullyReached":""}
  }
}

Rules:
- Header: "teacherName", "schoolName", "udise", "cycle" (1..5).
- For EACH student block, extract Kannada fields for name, SATS, caregiver name, student phone, caregiver phone, and grade. Phones: digits-only.
- Baseline & Endline highest operation:
  - Look ONLY for a drawn CIRCLE or a TICK MARK over the printed options line.
  - If you cannot clearly see exactly one circled/ticked option, set all values in baselineMarks/endlineMarks to false and leave baselineOp/endlineOp = "".
  - DO NOT infer from nearby words, letters, or context. Absence of a visible circle/tick => blank.
  - Valid values ONLY: "Beginner","Addition","Subtraction","Multiplication","Division".
- Kannada → English mapping for ops: ಆರಂಭಿಕ→"None"; ಸಂಕಲನ→"Addition"; ವಿಯೋಗ/ವಿವಕಲನ→"Subtraction"; ಗುಣಾಕಾರ→"Multiplication"; ಭಾಗಾಕಾರ→"Division".
- Sessions (4 rows): "Lesson 1".."Lesson 4"; datetime "DD/MM/YY HH:mm" (24h). If only date or time present, include what you see.
  "checkpointCorrect": "Yes"/"No"/""; "parentAttended": "Yes"/"No"/"".
- Footer: Lesson 1–4 with "totalStudents" and "successfullyReached".
- If a field is empty or illegible, output "".
- Return **valid JSON** and **nothing else**.
"""

DIGITS_ONLY_PROMPT = """
You see an image of a single form field that contains only a handwritten ID/phone number.
Return ONLY its digits as a single string (0-9). If unreadable, return "".
"""

# ====================== OPENAI HELPERS ======================
def make_client(key: str):
    from openai import OpenAI
    return OpenAI(api_key=key)

def data_url_from_bytes(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def call_json_vision(client, model: str, prompt: str, jpeg_bytes: bytes) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise JSON-only extractor. Output valid JSON only."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url_from_bytes(jpeg_bytes)}},
            ]}
        ]
    )
    return resp.choices[0].message.content

def call_digits_only(client, model: str, crop_bytes: bytes) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return JSON like {\"digits\":\"...\"} with digits only."},
            {"role": "user", "content": [
                {"type": "text", "text": DIGITS_ONLY_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url_from_bytes(crop_bytes)}},
            ]}
        ]
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt).get("digits","")
    except Exception:
        return re.sub(r"\D", "", txt)

def safe_json_loads(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}")+1]
    return json.loads(t)

# ====================== IMAGE PREPROCESS ======================
def preprocess_for_ocr(file) -> Tuple[bytes, np.ndarray, np.ndarray]:
    """
    Returns (jpeg_bytes, bin_gray_image, color_image_for_overlay)
    """
    raw = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    # 1) detect page and warp to top-down
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        page = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(page, True)
        approx = cv2.approxPolyDP(page, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            s = pts.sum(axis=1); diff = np.diff(pts, axis=1).ravel()
            ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
            tl, tr, br, bl = ordered
            widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
            maxW = int(max(widthA, widthB)); maxH = int(max(heightA, heightB))
            M = cv2.getPerspectiveTransform(ordered, np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32))
            img = cv2.warpPerspective(img, M, (maxW, maxH))

    color = img.copy()

    # 2) enhance + binarize
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

    # 3) encode JPEG
    pil = Image.fromarray(th)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), th, color

# ====================== CV MARK DETECTION ======================
def pick_op_by_ink(block_bin: np.ndarray, op_boxes: List[Tuple[str,float,float,float,float]], thresh: float) -> str:
    h, w = block_bin.shape[:2]
    best_label, best_ink = "", 0.0
    for label, x0, y0, x1, y1 in op_boxes:
        rx0, ry0, rx1, ry1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
        roi = block_bin[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            continue
        ink_ratio = 1.0 - (roi.mean()/255.0)  # darker => more ink
        if ink_ratio > best_ink:
            best_label, best_ink = label, ink_ratio
    return best_label if best_ink >= thresh else ""

def compute_cv_ops(bin_img: np.ndarray, overlay_img: np.ndarray, thresh: float, show_overlay: bool):
    H, W = bin_img.shape[:2]
    cv_ops = {}
    vis = overlay_img.copy()

    for idx, (yf0, yf1) in enumerate(STUDENT_STRIP_FRACS, start=1):
        y0, y1 = int(yf0*H), int(yf1*H)
        block = bin_img[y0:y1, :]
        base = pick_op_by_ink(block, BASELINE_OP_BOXES, thresh)
        end = pick_op_by_ink(block, ENDLINE_OP_BOXES, thresh)
        cv_ops[idx] = {"baselineOp": base or "", "endlineOp": end or ""}

        if show_overlay:
            # draw student strip
            cv2.rectangle(vis, (0, y0), (W-1, y1), (0, 255, 0), 2)
            # draw op boxes
            for (label, x0, y0f, x1, y1f) in BASELINE_OP_BOXES + ENDLINE_OP_BOXES:
                bx0, by0 = int(x0*W), int((y0f*(y1-y0) + y0))
                bx1, by1 = int(x1*W), int((y1f*(y1-y0) + y0))
                cv2.rectangle(vis, (bx0, by0), (bx1, by1), (255, 0, 0), 1)
                if label == base or label == end:
                    cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 0, 255), 2)

    return cv_ops, vis

# ====================== UTILITIES ======================
def digits_only(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())

def normalize_student_numbers(s: Dict[str, Any]) -> Dict[str, Any]:
    s["sats"] = digits_only(s.get("sats",""))
    s["studentPhone"] = digits_only(s.get("studentPhone",""))
    s["caregiverPhone"] = digits_only(s.get("caregiverPhone",""))
    return s

def sanitize_spreadsheet_id(text: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", text)
    return m.group(1) if m else text.split("?")[0].split("#")[0].strip()

def footer_to_df(obj: Dict[str, Any], file_name: str) -> pd.DataFrame:
    f = obj.get("footer",{}) or {}
    recs = []
    for i, key in enumerate(["lesson1","lesson2","lesson3","lesson4"], start=1):
        v = f.get(key, {}) or {}
        recs.append({
            "fileName": file_name,
            "lesson": f"Lesson {i}",
            "totalStudents": str(v.get("totalStudents","") or ""),
            "successfullyReached": str(v.get("successfullyReached","") or "")
        })
    return pd.DataFrame(recs)

def flatten_rows(obj: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    teacher = obj.get("teacherName","") or ""
    school  = obj.get("schoolName","") or ""
    udise   = digits_only(obj.get("udise","") or "")
    cycle   = obj.get("cycle","") or ""
    students = obj.get("students",[]) or []

    for s in students:
        s = normalize_student_numbers(s)
        baseline_op = s.get("baselineOp","") if s.get("baselineOp","") in ALLOWED_OPS else ""
        endline_op  = s.get("endlineOp","")  if s.get("endlineOp","")  in ALLOWED_OPS else ""

        base = {
            "fileName": file_name,
            "teacherName": teacher,
            "schoolName": school,
            "udise": udise,
            "cycle": cycle,
            "studentName": s.get("name","") or "",
            "sats": s.get("sats","") or "",
            "caregiverName": s.get("caregiverName","") or "",
            "studentPhone": s.get("studentPhone","") or "",
            "caregiverPhone": s.get("caregiverPhone","") or "",
            "grade": s.get("grade","") or "",
            "baselineOp": baseline_op,
            "endlineOp": endline_op,
        }

        sessions = s.get("sessions",[]) or []
        wrote_any = False
        for idx, sess in enumerate(sessions, start=1):
            if not any((sess.get("datetime"), sess.get("currentTopic"),
                        sess.get("checkpointCorrect"), sess.get("parentAttended"), sess.get("nextTopic"))):
                continue
            row = {**base,
                   "session": f"Lesson {idx}",
                   "datetime": sess.get("datetime","") or "",
                   "currentTopic": sess.get("currentTopic","") or "",
                   "checkpointCorrect": sess.get("checkpointCorrect","") or "",
                   "parentAttended": sess.get("parentAttended","") or "",
                   "nextTopic": sess.get("nextTopic","") or ""}
            rows.append(row)
            wrote_any = True

        if not wrote_any:
            rows.append({**base, "session":"", "datetime":"", "currentTopic":"",
                         "checkpointCorrect":"", "parentAttended":"", "nextTopic":""})

    for r in rows:
        for col in ROW_COLUMNS:
            r.setdefault(col,"")
    return rows

# Sheets helpers
def get_sheets_client():
    if not HAS_SHEETS:
        raise RuntimeError("gspread not installed.")
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Service account JSON missing in secrets as 'gcp_service_account'.")
    info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"],
    )
    return gspread.authorize(creds)

def append_df(ws, df: pd.DataFrame):
    values = df.astype(str).fillna("").values.tolist()
    if ws.row_count < len(values) + 1:
        ws.add_rows(len(values) + 10)
    ws.append_rows(values, value_input_option="USER_ENTERED")

def write_rows_to_sheets(df_rows: pd.DataFrame, spreadsheet_id: str, tab: str):
    gc = get_sheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(tab)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=len(df_rows.columns)+2)
        ws.append_row(df_rows.columns.tolist())
    append_df(ws, df_rows)

def write_footer_to_summary(df_footer: pd.DataFrame, spreadsheet_id: str):
    gc = get_sheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    tab = "summary"
    cols = ["fileName","lesson","totalStudents","successfullyReached"]
    try:
        ws = sh.worksheet(tab)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=len(cols)+2)
        ws.append_row(cols)
    if list(df_footer.columns) != cols:
        df_footer = df_footer[cols]
    append_df(ws, df_footer)

# ====================== MAIN ======================
if uploaded:
    if len(uploaded) > 10:
        st.error("Please upload 10 or fewer JPEG files.")
        st.stop()
    if not api_key:
        st.warning("Enter your OpenAI API key to start.")
        st.stop()

    from openai import OpenAI
    client = make_client(api_key)

    progress = st.progress(0)
    status = st.empty()

    all_rows: List[Dict[str, Any]] = []
    all_footer = []
    failures: List[str] = []
    debug_images = []

    for i, f in enumerate(uploaded, start=1):
        status.info(f"Processing {f.name} ({i}/{len(uploaded)}) ...")
        try:
            # Preprocess
            jpeg_bytes, bin_img, overlay_img = preprocess_for_ocr(f)

            # Deterministic circled/ticked detection (before model)
            cv_ops, vis = compute_cv_ops(bin_img, overlay_img, thresh=mark_thresh, show_overlay=show_overlay)
            if show_overlay:
                debug_images.append((f.name, vis[:, :, ::-1]))  # BGR->RGB

            # Call model for main JSON
            raw = call_json_vision(client, model=model, prompt=EXTRACTION_PROMPT, jpeg_bytes=jpeg_bytes)
            obj = safe_json_loads(raw)

            # Override ops with CV results (never guess)
            for idx, stu in enumerate(obj.get("students", []), start=1):
                if idx in cv_ops:
                    for which in ["baselineOp","endlineOp"]:
                        label = cv_ops[idx][which]
                        if label:
                            stu[which] = label
                            marks = {k:(k==label) for k in ALLOWED_OPS}
                        else:
                            stu[which] = ""
                            marks = {k: False for k in ALLOWED_OPS}
                        if which == "baselineOp":
                            stu["baselineMarks"] = marks
                        else:
                            stu["endlineMarks"] = marks

            # Optional extra digit pass (crops)
            if extra_digit_pass:
                H, W = bin_img.shape[:2]
                # Header UDISE
                x0, y0, x1, y1 = HEADER_UDISE_BOX
                crop = bin_img[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
                if crop.size > 0:
                    buff = io.BytesIO()
                    Image.fromarray(crop).save(buff, format="JPEG", quality=95)
                    try:
                        digits = call_digits_only(client, model, buff.getvalue())
                        if digits:
                            obj["udise"] = digits
                    except Exception:
                        pass

                # Per-student digits
                for idx, (yf0, yf1) in enumerate(STUDENT_STRIP_FRACS, start=1):
                    if idx-1 < len(obj.get("students", [])):
                        stu = obj["students"][idx-1]
                        y0p, y1p = int(yf0*H), int(yf1*H)
                        for label, (x0, y0, x1, y1) in [
                            ("sats", SAT_BOX), ("studentPhone", STUDENT_PHONE), ("caregiverPhone", CAREGIVER_PHONE)
                        ]:
                            cx0, cy0 = int(x0*W), int(y0*(y1p-y0p) + y0p)
                            cx1, cy1 = int(x1*W), int(y1*(y1p-y0p) + y0p)
                            crop = bin_img[cy0:cy1, cx0:cx1]
                            if crop.size > 0:
                                buff = io.BytesIO()
                                Image.fromarray(crop).save(buff, format="JPEG", quality=95)
                                try:
                                    digits = call_digits_only(client, model, buff.getvalue())
                                    if digits:
                                        stu[label] = digits
                                except Exception:
                                    pass

            # Shape rows
            rows = flatten_rows(obj, f.name)
            all_rows.extend(rows)
            all_footer.append(footer_to_df(obj, f.name))

        except Exception as e:
            failures.append(f"{f.name}: {str(e)[:220]}")
        finally:
            progress.progress(int(i*100/len(uploaded)))
            time.sleep(0.05)

    status.empty()

    if failures:
        st.error("Some files failed to digitize:")
        st.write("\n".join(failures))

    if all_rows:
        df_rows = pd.DataFrame(all_rows)[ROW_COLUMNS]
        st.success(f"Digitized {len(df_rows)} row(s) from {len(uploaded)-len(failures)} file(s).")

        # ===== Human-in-the-loop editor =====
        st.subheader("Review & edit (only these columns are editable)")
        editable_cols = ["currentTopic","nextTopic","checkpointCorrect"]

        # Restrict select options
        op_col = st.column_config.SelectboxColumn(
            "Operation", options=EDIT_OPS, help="Choose the operation or leave blank"
        )
        yn_col = st.column_config.SelectboxColumn(
            "Checkpoint Correct", options=EDIT_YN, help="Choose Yes/No or leave blank"
        )

        edited_df = st.data_editor(
            df_rows,
            column_config={
                "currentTopic": op_col,
                "nextTopic": op_col,
                "checkpointCorrect": yn_col
            },
            disabled=[c for c in df_rows.columns if c not in editable_cols],
            use_container_width=True,
            hide_index=True,
            key="editor",
        )

        # Normalize edited values (case/aliases)
        def norm_op(v: str) -> str:
            v = (v or "").strip().lower()
            m = {
                "addition":"Addition","add":"Addition","+":"Addition","a":"Addition",
                "subtraction":"Subtraction","minus":"Subtraction","-":"Subtraction","s":"Subtraction",
                "multiplication":"Multiplication","times":"Multiplication","x":"Multiplication","m":"Multiplication",
                "division":"Division","divide":"Division","/":"Division","d":"Division",
                "":""
            }
            return m.get(v, v.title() if v.title() in EDIT_OPS else "")

        def norm_yn(v: str) -> str:
            v = (v or "").strip().lower()
            if v in ("yes","y","true","1"): return "Yes"
            if v in ("no","n","false","0"): return "No"
            return ""

        edited_df["currentTopic"] = edited_df["currentTopic"].apply(norm_op)
        edited_df["nextTopic"]     = edited_df["nextTopic"].apply(norm_op)
        edited_df["checkpointCorrect"] = edited_df["checkpointCorrect"].apply(norm_yn)

        st.download_button("Download CSV (edited rows)",
                           edited_df.to_csv(index=False).encode("utf-8"),
                           "kannada_rows_edited.csv","text/csv")

        # Sheets push button (explicit)
        col_a, col_b = st.columns(2)
        with col_a:
            push_btn = st.button("Push EDITED rows to Google Sheet", type="primary")
        with col_b:
            push_summary_btn = st.button("Push summary to 'summary' tab")

        if push_btn:
            try:
                sid = sanitize_spreadsheet_id(spreadsheet_id_input)
                write_rows_to_sheets(edited_df, spreadsheet_id=sid, tab=worksheet_name)
                st.success(f"Appended {len(edited_df)} edited rows to '{worksheet_name}'.")
            except Exception as e:
                st.error(f"Google Sheets append failed: {e}")

        # Footer (unchanged, optional)
        if all_footer:
            df_footer = pd.concat(all_footer, ignore_index=True)
            with st.expander("Bottom summary (lesson-wise)"):
                st.dataframe(df_footer, use_container_width=True)

            if push_summary_btn:
                try:
                    sid = sanitize_spreadsheet_id(spreadsheet_id_input)
                    write_footer_to_summary(df_footer, spreadsheet_id=sid)
                    st.success("Appended summary rows to 'summary' tab.")
                except Exception as e:
                    st.error(f"Google Sheets append failed: {e}")

    else:
        st.warning("No rows extracted. Check image quality and that it matches the Kannada template.")

    # Debug overlays
    if show_overlay and debug_images:
        st.subheader("Detection overlays")
        for name, img_rgb in debug_images:
            st.image(img_rgb, caption=name, use_column_width=True)

else:
    st.info("Upload JPEG files to begin.")
