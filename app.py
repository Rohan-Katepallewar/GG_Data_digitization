import os, io, json, base64, time, re
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
from PIL import Image

# Optional Sheets support
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except Exception:
    HAS_SHEETS = False

# ----------------- App config -----------------
st.set_page_config(page_title="Kannada Paper Tool Digitizer", page_icon="🧾", layout="centered")
st.title("🧾 Kannada Paper Tool → JSON / CSV / Google Sheets")

st.markdown("""
Upload up to **10 JPEGs** of the fixed Kannada tutoring form.
Extracts header (**Teacher, School, UDISE, Cycle**), per-student details (**SATS, caregiver name/phone, grade**), **baseline/endline ops** (circled/ticked), **4 lessons**, and a bottom **summary**.

- Phones: digits-only, **any length**
- Baseline/Endline: **circled or tick-marked only** (no handwriting next to options)
- Processing is **sequential**; per-file failures are shown clearly
""")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Kept only in this session")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    # You can change these defaults anytime in the UI
    default_sheet = "16fnfCvrUBysEXtf_OAWLW1aVQSAegqtxehxwkWF3q_0"
    spreadsheet_id_input = st.text_input("Spreadsheet ID or URL (optional)", value=default_sheet)
    worksheet_name = st.text_input("Worksheet/Tab (rows)", value="sheet1")

    write_rows = st.checkbox("Append student-session rows to Google Sheet", value=False)
    write_footer = st.checkbox("Append bottom summary to 'summary' tab", value=False)

uploaded = st.file_uploader("Upload JPEG files (max 10)", type=["jpg","jpeg"], accept_multiple_files=True)

# ----------------- Prompt -----------------
# Kannada-focused: asks for Kannada OCR but returns English-keyed JSON
EXTRACTION_PROMPT = """
You are an OCR + information extraction agent for **Kannada** handwriting. The form layout is fixed (4 student blocks).

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
- Header: "teacherName", "schoolName", "udise", "cycle" (cycle is 1..5).
- For EACH student block, extract:
  - name (ಹೆಸರು), SATS (SATS #), caregiverName (ಪೋಷಕರ ಹೆಸರು if present),
  - studentPhone (ದೂರವಾಣಿ ಸಂಖ್ಯೆ), caregiverPhone (ಹೆಚ್ಚುವರಿ ದೂರವಾಣಿ ಸಂಖ್ಯೆ / ಪೋಷಕರ ದೂರವಾಣಿ),
  - grade (ತರಗತಿ).
- Baseline & Endline highest operation:
  - Choose ONLY the option that is **circled or tick-marked** (ignore free handwriting).
  - Valid values ONLY: "None","Addition","Subtraction","Multiplication","Division".
  - Kannada → English mapping:
    - ಆರಂಭಿಕ → "None"
    - ಸಂಕಲನ → "Addition"
    - ವಿಯೋಗ / ವಿಭಜನೆ / ವಿವಕಲನ (subtract) → "Subtraction"
    - ಗುಣಾಕಾರ → "Multiplication"
    - ಭಾಗಾಕಾರ → "Division"
  - In "baselineMarks" and "endlineMarks", set **true** only for the selected option(s). If none/unclear, set all false.
  - "baselineOp"/"endlineOp" may repeat the chosen option; if none, leave "".
- Sessions (4 rows):
  - "session": "Lesson 1".."Lesson 4" in row order.
  - "datetime": combine date & time as "DD/MM/YY HH:mm" (24h). If one missing, include available part.
  - "currentTopic": copy the Kannada letter/abbrev in that column.
  - "checkpointCorrect": "Yes" if marked correct, "No" if marked incorrect, else "".
  - "parentAttended": "Yes" if marked that parent/caregiver attended (ಪೋಷಕರು?), "No" if marked not, else "".
  - "nextTopic": copy what is written in that column.
- Footer table at the bottom: for Lesson 1–4, extract numbers for "totalStudents" and "successfullyReached".
- Phones: digits only (strip spaces, dashes, country code). Any length.
- If a field is empty or illegible, output "".
- Output must be **valid JSON** and contain **only** the JSON object.
"""

# ----------------- OpenAI helpers -----------------
def make_client(key: str):
    from openai import OpenAI
    return OpenAI(api_key=key)

def to_data_url_jpeg(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def call_vision_json(client, model: str, prompt: str, jpeg_bytes: bytes) -> str:
    data_url = to_data_url_jpeg(jpeg_bytes)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content":"You are a precise JSON-only extractor. Output valid JSON only."},
            {"role":"user","content":[
                {"type":"text","text": prompt},
                {"type":"image_url","image_url":{"url": data_url}}
            ]}
        ]
    )
    return resp.choices[0].message.content

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}")+1]
    return json.loads(text)

# ----------------- Row shaping -----------------
ALLOWED_OPS = ["None","Addition","Subtraction","Multiplication","Division"]

ROW_COLUMNS = [
    "fileName","teacherName","schoolName","udise","cycle",
    "studentName","sats","caregiverName","studentPhone","caregiverPhone","grade",
    "baselineOp","session","datetime","currentTopic","checkpointCorrect","parentAttended","nextTopic","endlineOp"
]

def pick_from_marks(m: dict) -> str:
    if not isinstance(m, dict):
        return ""
    trues = [k for k, v in m.items() if isinstance(v, bool) and v]
    return trues[0] if len(trues) == 1 else ""

def flatten_rows(obj: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    teacher = obj.get("teacherName","") or ""
    school  = obj.get("schoolName","") or ""
    udise   = obj.get("udise","") or ""
    cycle   = obj.get("cycle","") or ""
    students = obj.get("students",[]) or []

    for s in students:
        derived_baseline = pick_from_marks(s.get("baselineMarks", {}))
        derived_endline  = pick_from_marks(s.get("endlineMarks", {}))
        baseline_str = s.get("baselineOp","") or ""
        endline_str  = s.get("endlineOp","") or ""

        baseline_op = derived_baseline or (baseline_str if baseline_str in ALLOWED_OPS else "")
        endline_op  = derived_endline  or (endline_str  if endline_str  in ALLOWED_OPS else "")

        base = {
            "fileName": file_name,
            "teacherName": teacher,
            "schoolName": school,
            "udise": udise,
            "cycle": cycle,
            "studentName": s.get("name","") or "",
            "sats": s.get("sats","") or "",
            "caregiverName": s.get("caregiverName","") or "",
            "studentPhone": "".join(ch for ch in (s.get("studentPhone","") or "") if ch.isdigit()),
            "caregiverPhone": "".join(ch for ch in (s.get("caregiverPhone","") or "") if ch.isdigit()),
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

        if not wrote_any:  # write a baseline-only row so student isn't lost
            rows.append({**base, "session":"", "datetime":"", "currentTopic":"", "checkpointCorrect":"", "parentAttended":"", "nextTopic":""})

    # keep column order
    for r in rows:
        for col in ROW_COLUMNS:
            r.setdefault(col, "")
    return rows

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

# ----------------- Google Sheets -----------------
def sanitize_spreadsheet_id(text: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", text)
    return m.group(1) if m else text.split("?")[0].split("#")[0].strip()

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

# ----------------- Image utils -----------------
def ensure_jpeg(file) -> bytes:
    img = Image.open(file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

# ----------------- Main -----------------
if uploaded:
    if len(uploaded) > 10:
        st.error("Please upload 10 or fewer JPEG files.")
        st.stop()
    if not api_key:
        st.warning("Enter your OpenAI API key to start.")
        st.stop()

    from openai import OpenAI
    client = make_client(api_key)

    prog = st.progress(0)
    status = st.empty()

    all_rows: List[Dict[str, Any]] = []
    all_footer = []
    failures: List[str] = []

    for i, f in enumerate(uploaded, start=1):
        status.info(f"Processing {f.name} ({i}/{len(uploaded)}) ...")
        try:
            jpeg = ensure_jpeg(f)
            raw = call_vision_json(client, model=model, prompt=EXTRACTION_PROMPT, jpeg_bytes=jpeg)
            obj = safe_json_loads(raw)
            rows = flatten_rows(obj, f.name)
            all_rows.extend(rows)
            all_footer.append(footer_to_df(obj, f.name))
        except Exception as e:
            failures.append(f"{f.name}: {str(e)[:220]}")
        finally:
            prog.progress(int(i * 100 / len(uploaded)))
            time.sleep(0.05)

    status.empty()

    if failures:
        st.error("Some files failed to digitize:")
        st.write("\n".join(failures))

    if all_rows:
        df_rows = pd.DataFrame(all_rows)[ROW_COLUMNS]
        st.success(f"Digitized {len(df_rows)} row(s) from {len(uploaded)-len(failures)} file(s).")
        st.dataframe(df_rows, use_container_width=True)

        st.download_button("Download CSV (rows)",
                           df_rows.to_csv(index=False).encode("utf-8"),
                           "kannada_rows.csv","text/csv")

        if all_footer:
            df_footer = pd.concat(all_footer, ignore_index=True)
            with st.expander("Bottom summary (lesson-wise)"):
                st.dataframe(df_footer, use_container_width=True)
            st.download_button("Download CSV (summary)",
                               df_footer.to_csv(index=False).encode("utf-8"),
                               "kannada_summary.csv","text/csv")
        else:
            df_footer = pd.DataFrame(columns=["fileName","lesson","totalStudents","successfullyReached"])

        # Sheets
        if write_rows or write_footer:
            try:
                sid = sanitize_spreadsheet_id(spreadsheet_id_input)
                if write_rows:
                    write_rows_to_sheets(df_rows, spreadsheet_id=sid, tab=worksheet_name)
                    st.success(f"Appended {len(df_rows)} rows to '{worksheet_name}'.")
                if write_footer and not df_footer.empty:
                    write_footer_to_summary(df_footer, spreadsheet_id=sid)
                    st.success("Appended summary rows to 'summary' tab.")
            except Exception as e:
                st.error(f"Google Sheets append failed: {e}")
    else:
        st.warning("No rows extracted. Check image quality and ensure it matches the Kannada template.")
else:
    st.info("Upload JPEG files to begin.")
