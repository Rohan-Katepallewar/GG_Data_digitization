import io, os, re, json, base64, time, hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---- Try OpenCV; fallback if unavailable ----
try:
    import cv2
    CV_OK = True
except Exception:
    CV_OK = False

# ---- Optional Google Sheets ----
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except Exception:
    HAS_SHEETS = False

# ====================== STREAMLIT UI CONFIG ======================
st.set_page_config(page_title="Kannada Paper Tool Digitizer (HIL, stable)", page_icon="🧾", layout="wide")
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
    mark_thresh = st.slider("Circled baseline/endline ink threshold", 0.05, 0.30, 0.12, 0.01)
    extra_digit_pass = st.checkbox("Extra digit pass (SATS/phones/UDISE)", value=False)
    extra_ops_pass = st.checkbox("Extra ops pass (Current/Next via cell crops)", value=True)
    show_overlay = st.checkbox("Show debug overlay (ops + circled boxes)", value=False)

uploaded = st.file_uploader("Upload JPEG files (max 10)", type=["jpg","jpeg"], accept_multiple_files=True)

# ==== ACTION BUTTONS (prevents reprocessing while editing) ====
colX, colY = st.columns([1,1])
with colX:
    run_btn = st.button("Process files", type="primary", use_container_width=True)
with colY:
    rerun_btn = st.button("Re-process with current settings", use_container_width=True)

# ====================== CONSTANTS ======================
ALLOWED_OPS = ["None","Addition","Subtraction","Multiplication","Division"]
EDIT_OPS = ["", "Addition","Subtraction","Multiplication","Division"]
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

# Within each student strip, fractional boxes for the 5 circled ops (baseline row band / endline row band)
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

# ---- Ops cell crops (TUNE ONCE using overlay) ----
# For each student strip, four lesson row bands (fractions of strip height)
LESSON_ROWS = [
    (0.26, 0.34),   # Lesson 1 row band
    (0.36, 0.44),   # Lesson 2
    (0.46, 0.54),   # Lesson 3
    (0.56, 0.64),   # Lesson 4
]
# Column windows inside strip for ops cells
CURRENT_COL = (0.42, 0.53)   # Current Topic cell (x0, x1) within strip, tune if needed
NEXT_COL    = (0.70, 0.86)   # Next Topic cell (x0, x1) within strip, tune if needed

# Header crop for UDISE (fractions of full page)
HEADER_UDISE_BOX = (0.75, 0.03, 0.96, 0.08)

# Per-student small crops within strip for digits pass
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
  - Valid values ONLY: "None","Addition","Subtraction","Multiplication","Division".
- Sessions (4 rows) come from a table with columns for date/time, current topic, checkpoint, parent attended, next topic.
  - "session": "Lesson 1".."Lesson 4"; "datetime": "DD/MM/YY HH:mm" (24h). If only date or only time is present, include what you see.
  - For "currentTopic" and "nextTopic": map any Kannada word/abbr, English word/abbr, single letter (A/S/M/D), or symbol (+/−/×/÷) to exactly:
      "Addition","Subtraction","Multiplication","Division".
    If the cell is blank or unreadable, use "" (do NOT assume any sequence).
  - "checkpointCorrect": "Yes" if marked correct, "No" if marked incorrect, else "".
  - "parentAttended": "Yes"/"No"/"".
- Footer: Lesson 1–4 with "totalStudents" and "successfullyReached".
- If a field is empty or illegible, output "".
- Return **valid JSON** and **nothing else**.
"""

DIGITS_ONLY_PROMPT = """
You see an image of a single form field that contains only a handwritten ID/phone number.
Return ONLY its digits as a single string (0-9). If unreadable, return "".
"""

OPS_ONLY_PROMPT = """
You see a small cropped cell from a tutoring log. If it contains a symbol/letter/word for a math operation, map to exactly one of:
"Addition","Subtraction","Multiplication","Division".
If the cell is blank or unclear, return "".
Return JSON like {"op":"Addition"} (or with "" if blank). Do NOT infer any sequence.
"""

# ====================== OPENAI HELPERS ======================
def make_client(key: str):
    from openai import OpenAI
    return OpenAI(api_key=key)

def data_url_from_bytes(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def call_json_vision(client, model: str, prompt: str, jpeg_bytes: bytes, timeout_sec: int = 45) -> str:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role":"system","content":"You are a precise JSON-only extractor. Output valid JSON only."},
                    {"role":"user","content":[
                        {"type":"text","text": prompt},
                        {"type":"image_url","image_url":{"url": data_url_from_bytes(jpeg_bytes)}}
                    ]}
                ],
                timeout=timeout_sec,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI vision call failed after retries: {last_err}")

def call_digits_only(client, model: str, crop_bytes: bytes, timeout_sec: int = 30) -> str:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return JSON like {\"digits\":\"...\"} with digits only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": DIGITS_ONLY_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url_from_bytes(crop_bytes)}},
                    ]}
                ],
                timeout=timeout_sec,
            )
            txt = resp.choices[0].message.content.strip()
            try:
                return json.loads(txt).get("digits","")
            except Exception:
                return re.sub(r"\D", "", txt)
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"OpenAI digits call failed after retries: {last_err}")

def call_ops_only(client, model: str, crop_bytes: bytes, timeout_sec: int = 25) -> str:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return JSON like {\"op\":\"Addition|Subtraction|Multiplication|Division|\"} (empty string allowed)."},
                    {"role": "user", "content": [
                        {"type": "text", "text": OPS_ONLY_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url_from_bytes(crop_bytes)}},
                    ]}
                ],
                timeout=timeout_sec,
            )
            txt = resp.choices[0].message.content.strip()
            try:
                op = json.loads(txt).get("op","")
            except Exception:
                # soft fallback: try to pick token
                m = re.search(r"(Addition|Subtraction|Multiplication|Division)", txt, re.I)
                op = m.group(1).title() if m else ""
            return op
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"OpenAI ops call failed after retries: {last_err}")

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
def preprocess_for_ocr_bytes(file_bytes: bytes) -> Tuple[bytes, np.ndarray, np.ndarray]:
    """
    Returns (api_jpeg_bytes_color_small, bin_gray_image, color_image_for_overlay)
    - Sends small COLOR JPEG to OpenAI (faster, clearer)
    - Keeps binarized image for CV mark detection / overlays
    """
    pil_in = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    np_in = np.array(pil_in)[:, :, ::-1]  # RGB->BGR

    if CV_OK:
        img = np_in.copy()
        # Try perspective warp to top-down
        try:
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
        except Exception:
            pass

        color = img.copy()

        # Enhance + binarize
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g_enh = clahe.apply(g)
        th = cv2.adaptiveThreshold(g_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

        # Resize color for API (max side 1600)
        h, w = color.shape[:2]
        max_side = 1600
        scale = min(1.0, max_side / max(h, w))
        color_small = cv2.resize(color, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else color

        pil_small = Image.fromarray(cv2.cvtColor(color_small, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_small.save(buf, format="JPEG", quality=80)
        api_bytes = buf.getvalue()
        return api_bytes, th, color

    else:
        img = pil_in
        max_side = 1600
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        img_small = img.resize((int(w*scale), int(h*scale))) if scale < 1.0 else img
        buf = io.BytesIO()
        img_small.save(buf, format="JPEG", quality=80)
        api_bytes = buf.getvalue()
        gray = np.array(img.convert("L"))
        th = (gray > 200).astype("uint8") * 255
        color = np.array(img)[:, :, ::-1]
        return api_bytes, th, color

# ====================== CV HELPERS ======================
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
    if not CV_OK:
        return {}, overlay_img
    H, W = bin_img.shape[:2]
    cv_ops = {}
    vis = overlay_img.copy()

    for idx, (yf0, yf1) in enumerate(STUDENT_STRIP_FRACS, start=1):
        y0, y1 = int(yf0*H), int(yf1*H)
        block = bin_img[y0:y1, :]
        base = pick_op_by_ink(block, BASELINE_OP_BOXES, thresh)
        end = pick_op_by_ink(block, ENDLINE_OP_BOXES, thresh)
        cv_ops[idx] = {"baselineOp": base or "", "endlineOp": end or ""}

        if show_overlay and CV_OK:
            cv2.rectangle(vis, (0, y0), (W-1, y1), (0, 255, 0), 2)
            for (label, x0, y0f, x1, y1f) in BASELINE_OP_BOXES + ENDLINE_OP_BOXES:
                bx0, by0 = int(x0*W), int((y0f*(y1-y0) + y0))
                bx1, by1 = int(x1*W), int((y1f*(y1-y0) + y0))
                cv2.rectangle(vis, (bx0, by0), (bx1, by1), (255, 0, 0), 1)
                if label == base or label == end:
                    cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 0, 255), 2)

    return cv_ops, vis

def crop_ops_cells(bin_img: np.ndarray, show_overlay_img: np.ndarray) -> Tuple[Dict[Tuple[int,int,str], bytes], np.ndarray]:
    """
    Returns dict keyed by (student_idx, lesson_idx, which in {'current','next'}) -> JPEG bytes of the cell.
    """
    H, W = bin_img.shape[:2]
    crops: Dict[Tuple[int,int,str], bytes] = {}
    vis = show_overlay_img.copy()

    for s_idx, (yf0, yf1) in enumerate(STUDENT_STRIP_FRACS, start=1):
        y0, y1 = int(yf0*H), int(yf1*H)
        strip_h = y1 - y0

        for l_idx, (rf0, rf1) in enumerate(LESSON_ROWS, start=1):
            ry0 = int(y0 + rf0*strip_h)
            ry1 = int(y0 + rf1*strip_h)

            # Current cell
            cx0 = int(CURRENT_COL[0]*W)
            cx1 = int(CURRENT_COL[1]*W)
            cur = bin_img[ry0:ry1, cx0:cx1]

            # Next cell
            nx0 = int(NEXT_COL[0]*W)
            nx1 = int(NEXT_COL[1]*W)
            nxt = bin_img[ry0:ry1, nx0:nx1]

            for which, cell in [("current", cur), ("next", nxt)]:
                if cell.size == 0:
                    continue
                # If nearly blank, skip API
                ink_ratio = 1.0 - (cell.mean()/255.0)
                if ink_ratio < 0.01:
                    # store empty JPEG (so we don't call API)
                    buf = io.BytesIO(); Image.fromarray(cell).save(buf, format="JPEG", quality=80)
                    crops[(s_idx, l_idx, which)] = b""  # mark as blank
                else:
                    buf = io.BytesIO()
                    Image.fromarray(cell).save(buf, format="JPEG", quality=85)
                    crops[(s_idx, l_idx, which)] = buf.getvalue()

            if show_overlay and CV_OK:
                # draw rectangles
                cv2.rectangle(vis, (cx0, ry0), (cx1, ry1), (200, 0, 200), 2)  # current
                cv2.rectangle(vis, (nx0, ry0), (nx1, ry1), (0, 128, 255), 2)  # next
                cv2.putText(vis, f"S{s_idx}L{l_idx}", (cx0, max(ry0-4,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,0,200), 1, cv2.LINE_AA)

    return crops, vis

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

# ---- Canonicalization helpers (for model outputs + human edits) ----
CANONICAL_OPS = {"Addition", "Subtraction", "Multiplication", "Division"}
_OP_MAP = {
    # Kannada full words
    "ಸಂಕಲನ": "Addition", "ಜೋಡಣೆ": "Addition",
    "ವಿಯೋಗ": "Subtraction", "ವಿವಕಲನ": "Subtraction", "ಕಳೆಯುವುದು": "Subtraction",
    "ಗುಣಾಕಾರ": "Multiplication",
    "ಭಾಗಾಕಾರ": "Division", "ಭಾಗ": "Division",
    # English words/abbrevs
    "addition":"Addition","add":"Addition","sum":"Addition","plus":"Addition",
    "subtraction":"Subtraction","subtract":"Subtraction","minus":"Subtraction","sub":"Subtraction",
    "multiplication":"Multiplication","multiply":"Multiplication","times":"Multiplication","mul":"Multiplication",
    "division":"Division","divide":"Division","div":"Division",
    # single letters
    "a":"Addition","s":"Subtraction","m":"Multiplication","d":"Division",
    # symbols
    "+":"Addition","-":"Subtraction","×":"Multiplication","x":"Multiplication","*":"Multiplication","÷":"Division","/":"Division",
    # short-hands
    "addn":"Addition","subs":"Subtraction"
}
def coerce_op(val: str) -> str:
    if not val:
        return ""
    v = str(val).strip().lower()
    if v in _OP_MAP:
        return _OP_MAP[v]
    v2 = re.sub(r"[^a-z+\-x*/×÷/ಅ-ಹ]", "", v)
    if v2 in _OP_MAP:
        return _OP_MAP[v2]
    vt = v.title()
    return vt if vt in CANONICAL_OPS else ""
def norm_op(v: str) -> str:
    return coerce_op(v)
def norm_yn(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("yes","y","true","1"): return "Yes"
    if v in ("no","n","false","0"): return "No"
    return ""

# ====================== ROW SHAPING ======================
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
                   "currentTopic": coerce_op(sess.get("currentTopic","") or ""),
                   "checkpointCorrect": sess.get("checkpointCorrect","") or "",
                   "parentAttended": sess.get("parentAttended","") or "",
                   "nextTopic": coerce_op(sess.get("nextTopic","") or "")}
            rows.append(row)
            wrote_any = True

        if not wrote_any:
            rows.append({**base, "session":"", "datetime":"", "currentTopic":"",
                         "checkpointCorrect":"", "parentAttended":"", "nextTopic":""})

    for r in rows:
        for col in ROW_COLUMNS:
            r.setdefault(col,"")
    return rows

# ====================== SHEETS HELPERS ======================
@st.cache_resource
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

# ====================== PIPELINE (single run; returns cached results) ======================
def process_files_once(files_data: List[Tuple[str, bytes]],
                       api_key: str, model: str,
                       mark_thresh: float,
                       extra_digit_pass: bool,
                       extra_ops_pass: bool,
                       show_overlay: bool):
    from openai import OpenAI
    client = make_client(api_key)

    all_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    overlays = []

    for i, (fname, fbytes) in enumerate(files_data, start=1):
        try:
            api_jpeg_bytes, bin_img, overlay_img = preprocess_for_ocr_bytes(fbytes)

            # circled/ticked baseline/endline via CV
            cv_ops, vis = compute_cv_ops(bin_img, overlay_img, thresh=mark_thresh, show_overlay=show_overlay)

            # model JSON
            raw = call_json_vision(client, model=model, prompt=EXTRACTION_PROMPT, jpeg_bytes=api_jpeg_bytes)
            obj = safe_json_loads(raw)

            # override circled ops with CV
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

            H, W = bin_img.shape[:2]

            # optional: extra digits
            if extra_digit_pass:
                # header UDISE
                x0, y0, x1, y1 = HEADER_UDISE_BOX
                cx0, cy0, cx1, cy1 = int(x0*W), int(y0*H), int(x1*W), int(y1*H)
                crop = bin_img[cy0:cy1, cx0:cx1]
                if crop.size > 0:
                    buff = io.BytesIO(); Image.fromarray(crop).save(buff, format="JPEG", quality=85)
                    try:
                        obj["udise"] = call_digits_only(client, model, buff.getvalue()) or obj.get("udise","")
                    except Exception:
                        pass

                # per-student digits
                for s_idx, (yf0, yf1) in enumerate(STUDENT_STRIP_FRACS, start=1):
                    if s_idx-1 < len(obj.get("students", [])):
                        stu = obj["students"][s_idx-1]
                        y0p, y1p = int(yf0*H), int(yf1*H)
                        for label, (x0, y0f, x1, y1f) in [
                            ("sats", SAT_BOX), ("studentPhone", STUDENT_PHONE), ("caregiverPhone", CAREGIVER_PHONE)
                        ]:
                            cx0, cy0 = int(x0*W), int(y0f*(y1p-y0p) + y0p)
                            cx1, cy1 = int(x1*W), int(y1f*(y1p-y0p) + y0p)
                            crop = bin_img[cy0:cy1, cx0:cx1]
                            if crop.size > 0:
                                buff = io.BytesIO(); Image.fromarray(crop).save(buff, format="JPEG", quality=85)
                                try:
                                    digits = call_digits_only(client, model, buff.getvalue())
                                    if digits:
                                        stu[label] = digits
                                except Exception:
                                    pass

            # optional: ops per-cell
            if extra_ops_pass:
                crops, vis2 = crop_ops_cells(bin_img, overlay_img)
                if show_overlay and CV_OK:
                    # blend both overlays
                    try:
                        vis = cv2.addWeighted(vis, 1.0, vis2, 0.9, 0)
                    except Exception:
                        pass

                # fill current/next from crops (conservative; no sequence inference)
                for (s_idx, l_idx, which), jpg in crops.items():
                    if s_idx-1 >= len(obj.get("students", [])): 
                        continue
                    sess_list = obj["students"][s_idx-1].get("sessions", [])
                    if l_idx-1 >= len(sess_list):
                        continue
                    if jpg == b"":   # explicitly blank cell
                        op = ""
                    else:
                        try:
                            op = call_ops_only(client, model, jpg)
                        except Exception:
                            op = ""
                    op = coerce_op(op)
                    key = "currentTopic" if which == "current" else "nextTopic"
                    if op:
                        sess_list[l_idx-1][key] = op
                    else:
                        # leave model value if it had something; else blank
                        sess_list[l_idx-1][key] = coerce_op(sess_list[l_idx-1].get(key,""))

            rows = flatten_rows(obj, fname)
            all_rows.extend(rows)

            if show_overlay and isinstance(vis, np.ndarray):
                overlays.append((fname, vis[:, :, ::-1] if CV_OK else vis))

        except Exception as e:
            failures.append(f"{fname}: {str(e)[:220]}")

    return all_rows, overlays, failures

# ====================== MAIN ======================
def compute_cache_key(files_data, model, mark_thresh, extra_digit_pass, extra_ops_pass):
    h = hashlib.md5()
    h.update(model.encode())
    h.update(str(mark_thresh).encode())
    h.update(b"D" if extra_digit_pass else b"d")
    h.update(b"O" if extra_ops_pass else b"o")
    for name, data in files_data:
        h.update(name.encode()); h.update(str(len(data)).encode())
        h.update(hashlib.md5(data).digest())
    return h.hexdigest()

if uploaded:
    if len(uploaded) > 10:
        st.error("Please upload 10 or fewer JPEG files.")
        st.stop()
    if not api_key:
        st.warning("Enter your OpenAI API key to start.")
        st.stop()

    # Read files once to bytes (so edits don't re-run the pipeline)
    files_data = [(f.name, f.getvalue()) for f in uploaded]
    cache_key = compute_cache_key(files_data, model, mark_thresh, extra_digit_pass, extra_ops_pass)

    # Decide if we should process now
    should_process = False
    if run_btn or rerun_btn:
        should_process = True
    elif "cache_key" not in st.session_state:
        should_process = True
    elif st.session_state.get("cache_key") != cache_key:
        should_process = True

    if should_process:
        progress = st.progress(0)
        status = st.empty()
        all_rows, overlays, failures = process_files_once(
            files_data, api_key, model, mark_thresh, extra_digit_pass, extra_ops_pass, show_overlay
        )
        # stash results
        st.session_state["cache_key"] = cache_key
        st.session_state["rows_df"] = pd.DataFrame(all_rows)[ROW_COLUMNS] if all_rows else pd.DataFrame(columns=ROW_COLUMNS)
        st.session_state["overlays"] = overlays
        st.session_state["failures"] = failures
        st.session_state["edited_df"] = st.session_state["rows_df"].copy()
        status.empty()
        progress.progress(100)
        time.sleep(0.05)

    # Present cached results (no reprocessing during edit)
    failures = st.session_state.get("failures", [])
    if failures:
        st.error("Some files failed to digitize:")
        st.write("\n".join(failures))

    df_rows = st.session_state.get("rows_df", pd.DataFrame(columns=ROW_COLUMNS))
    if not df_rows.empty:
        st.success(f"Ready: {len(df_rows)} row(s) from {len(uploaded)-len(failures)} file(s).")
        st.subheader("Review & edit (editable: currentTopic, nextTopic, checkpointCorrect)")
        editable_cols = ["currentTopic","nextTopic","checkpointCorrect"]

        op_col = st.column_config.SelectboxColumn("Operation", options=EDIT_OPS, help="Choose operation or leave blank")
        yn_col = st.column_config.SelectboxColumn("Checkpoint Correct", options=EDIT_YN, help="Yes/No or blank")

        # start from previous edits if present
        base_df = st.session_state.get("edited_df", df_rows.copy())

        edited_df = st.data_editor(
            base_df,
            column_config={"currentTopic": op_col, "nextTopic": op_col, "checkpointCorrect": yn_col},
            disabled=[c for c in df_rows.columns if c not in editable_cols],
            use_container_width=True, hide_index=True, key="editor",
        )
        # Normalize human edits
        edited_df["currentTopic"] = edited_df["currentTopic"].apply(norm_op)
        edited_df["nextTopic"]    = edited_df["nextTopic"].apply(norm_op)
        edited_df["checkpointCorrect"] = edited_df["checkpointCorrect"].apply(norm_yn)

        # Save back to session (so further reruns keep edits)
        st.session_state["edited_df"] = edited_df.copy()

        st.download_button("Download CSV (edited rows)",
                           edited_df.to_csv(index=False).encode("utf-8"),
                           "kannada_rows_edited.csv","text/csv")

        # Push edited rows
        if st.button("Push EDITED rows to Google Sheet", type="primary"):
            try:
                sid = sanitize_spreadsheet_id(spreadsheet_id_input)
                write_rows_to_sheets(edited_df, spreadsheet_id=sid, tab=worksheet_name)
                st.success(f"Appended {len(edited_df)} edited rows to '{worksheet_name}'.")
            except Exception as e:
                st.error(f"Google Sheets append failed: {e}")

        # Overlays
        if show_overlay and st.session_state.get("overlays"):
            st.subheader("Detection overlays (tune boxes if needed)")
            for name, img_rgb in st.session_state["overlays"]:
                st.image(img_rgb, caption=name, use_column_width=True)

    else:
        st.warning("No rows extracted. Click 'Process files' to run.")

else:
    st.info("Upload JPEG files and click 'Process files'.")
