import re
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from datetime import date, datetime
from ultralytics import YOLO
import easyocr
import numpy as np

from database import initialize_db, add_owner, lookup_plate, get_all_owners, delete_owner, check_document_violations
from messenger import send_email, build_alert_message, build_alert_subject

# Helmet violation is added to the violations list passed to build_alert_message
HELMET_VIOLATION_MSG = "Rider detected without helmet (traffic safety violation)"

# ---------------------------------------------------------
# Initialise database on startup
# ---------------------------------------------------------
initialize_db()

# ---------------------------------------------------------
# Load YOLO Models
# ---------------------------------------------------------
# Plate detection — prefer latest Indian-plate fine-tuned model
_PLATE_MODEL_PRIORITY = [
    "models/indian-plate-detector/weights/best.pt",   # latest (Indian-trained)
    "models/license-plate-retrain/weights/best.pt",   # fallback
    "models/yolov12n-license-plate/weights/best.pt",  # oldest fallback
]
MODEL_PATH = next((p for p in _PLATE_MODEL_PRIORITY if os.path.exists(p)), _PLATE_MODEL_PRIORITY[-1])
print("Plate model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

HELMET_MODEL_PATH = "models/helmet-detector/weights/best.pt"
_helmet_model = None
if os.path.exists(HELMET_MODEL_PATH):
    _helmet_model = YOLO(HELMET_MODEL_PATH)

# COCO-pretrained model for vehicle detection (car, motorcycle, truck, bus)
VEHICLE_MODEL_PATH = "models/yolov8n.pt"
_vehicle_model = YOLO(VEHICLE_MODEL_PATH) if os.path.exists(VEHICLE_MODEL_PATH) else None
# COCO class IDs for vehicles we care about
_VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# ---------------------------------------------------------
# Load OCR
# ---------------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------------------------
# Preprocess cropped plate for better OCR accuracy
# ---------------------------------------------------------
def pad_crop(image, x1, y1, x2, y2, pad_frac=0.08):
    """Expand bounding box by pad_frac on each side, clamped to image bounds."""
    h, w = image.shape[:2]
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    return image[max(0, y1 - ph):min(h, y2 + ph),
                 max(0, x1 - pw):min(w, x2 + pw)]


def _deskew(img):
    """Rotate plate crop to correct skew detected via Hough lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180,
                           threshold=max(20, img.shape[1] // 5))
    if lines is None:
        return img
    angles = []
    for line in lines[:20]:
        _, theta = line[0]
        angle = np.degrees(theta) - 90
        if -25 < angle < 25:
            angles.append(angle)
    if not angles:
        return img
    median_angle = float(np.median(angles))
    if abs(median_angle) < 1.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_paths(plate_img, target_height=96):
    """
    Scale + deskew + return three preprocessed versions of the plate crop:
      0 – CLAHE enhanced + sharpened  (good for colour images)
      1 – Otsu binary threshold        (clean high-contrast plates)
      2 – Adaptive Gaussian threshold  (works under uneven lighting)
    """
    h, w = plate_img.shape[:2]
    if h < target_height:
        scale = target_height / h
        plate_img = cv2.resize(
            plate_img, (int(w * scale), target_height),
            interpolation=cv2.INTER_CUBIC,
        )
    plate_img = _deskew(plate_img)

    # --- Path 0: CLAHE + sharpen ---
    lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    p0 = cv2.filter2D(enhanced, -1, kernel)

    # --- Path 1: Otsu binary (on grayscale) ---
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, p1_g = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p1 = cv2.cvtColor(p1_g, cv2.COLOR_GRAY2BGR)

    # --- Path 2: Adaptive Gaussian threshold ---
    p2_g = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    p2 = cv2.cvtColor(p2_g, cv2.COLOR_GRAY2BGR)

    return [p0, p1, p2]


# ---------------------------------------------------------
# Extract text from cropped plate
# ---------------------------------------------------------
PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_LETTER_TO_DIGIT = str.maketrans("OIZBS", "01285")
_DIGIT_TO_LETTER = str.maketrans("01285", "OIZBS")

# Indian plate format patterns — (regex, positional-type-string)
# L=letter, D=digit
_PLATE_PATTERNS = [
    # Standard series: MH12AB1234
    (re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'),   "LLDDLLDDDD"),
    # 5-digit series: MH12AB12345
    (re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{5}$'),   "LLDDLLDDDDD"),
    # Single-letter series: MH12A1234
    (re.compile(r'^[A-Z]{2}\d{2}[A-Z]\d{4}$'),      "LLDDLDDDD"),
    # Three-letter series: MH12ABC1234
    (re.compile(r'^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$'),   "LLDDLLLDDDD"),
    # BH series: 22BH1234A or 22BH1234AA
    (re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$'),       None),
]
_NOISE_RE = re.compile(r'[\s\-\.\,\_]')


def _correct_plate_text(text):
    t = _NOISE_RE.sub("", text.upper().strip())
    if not t:
        return t

    # Try each known pattern and fix obvious OCR substitutions per position
    for pattern, pos_type in _PLATE_PATTERNS:
        if pos_type is None:
            # BH series — just validate, no character-type correction needed
            if pattern.match(t):
                return t
            continue
        if len(t) == len(pos_type):
            corrected = ""
            for ch, typ in zip(t, pos_type):
                if typ == 'L':
                    corrected += ch.translate(_DIGIT_TO_LETTER)
                else:
                    corrected += ch.translate(_LETTER_TO_DIGIT)
            return corrected

    # Fallback: at minimum correct positions 2-3 (should be digits after state code)
    if len(t) >= 4:
        return t[0:2] + t[2:4].translate(_LETTER_TO_DIGIT) + t[4:]
    return t


def _keep_plate_segments(results, crop_h, min_char_height_frac=0.15):
    kept = []
    for (bbox, txt, conf) in results:
        ys = [pt[1] for pt in bbox]
        seg_h = max(ys) - min(ys)
        if seg_h >= crop_h * min_char_height_frac:
            kept.append((bbox, txt, conf))
    return kept


def read_plate_text(plate_img, ocr_conf_thresh=0.2):
    """
    Try multiple preprocessed versions of the plate crop and keep whichever
    gives the highest-confidence OCR reading.
    """
    paths = _preprocess_paths(plate_img)
    best_results = []
    best_conf = -1.0

    for path_img in paths:
        crop_h = path_img.shape[0]
        res = reader.readtext(
            path_img, allowlist=PLATE_CHARS, paragraph=False, detail=1,
        )
        res = _keep_plate_segments(res, crop_h)
        conf = max((c for _, _, c in res), default=0.0)
        if conf > best_conf:
            best_conf = conf
            best_results = res

    good = [r for r in best_results if r[2] >= ocr_conf_thresh]
    # Sort top-to-bottom then left-to-right so multi-line plates read correctly
    good.sort(key=lambda r: (min(pt[1] for pt in r[0]), min(pt[0] for pt in r[0])))
    raw_text = "".join(txt for (_, txt, _conf) in good)
    return _correct_plate_text(raw_text)


# ---------------------------------------------------------
# Helmet / No-Helmet Detection
# ---------------------------------------------------------
# Model class names (hard-hat-workers dataset):
#   0 = head   → bare head = person present + VIOLATION
#   1 = helmet → helmeted  = person present, no violation
#   2 = person → body (recall ~1% – not used for person detection)
_NO_HELMET_NAMES = {"head", "no_helmet", "no-helmet", "without_helmet",
                    "without-helmet", "nohelmet"}
_HELMET_NAMES = {"helmet"}


def detect_helmet_violations(image, conf_thresh=0.35):
    """
    Run helmet detector on the full frame.

    A person is considered present on the vehicle when at least one
    'head' or 'helmet' bounding box is detected (the 'person' class
    has ~1 % recall and is ignored for this purpose).

    Returns:
        annotated_image  – frame with colour-coded boxes drawn
        person_detected  – True if any rider head/helmet found
        helmet_violation – True if any bare-head (no-helmet) detected
        no_helmet_count  – number of bare-head detections
    If the helmet model is not loaded, returns (image, False, False, 0).
    """
    if _helmet_model is None:
        return image, False, False, 0

    annotated = image.copy()
    results = _helmet_model(annotated, conf=conf_thresh)
    no_helmet_count = 0
    helmet_count = 0

    for r in results:
        for box, cls_id, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
        ):
            label = r.names[cls_id].lower()
            is_violation = label in _NO_HELMET_NAMES
            is_helmet = label in _HELMET_NAMES
            if is_violation:
                no_helmet_count += 1
            elif is_helmet:
                helmet_count += 1
            # Skip drawing 'person' class boxes – not useful here
            if not (is_violation or is_helmet):
                continue
            color = (0, 0, 220) if is_violation else (0, 200, 0)   # red / green
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = "NO HELMET" if is_violation else "Helmet OK"
            cv2.putText(
                annotated, f"{tag} {conf:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

    person_detected = (no_helmet_count + helmet_count) > 0
    return annotated, person_detected, no_helmet_count > 0, no_helmet_count


# ---------------------------------------------------------
# Plate Detection + OCR
# ---------------------------------------------------------
def _get_vehicle_rois(image, conf_thresh=0.35, pad_frac=0.05):
    """
    Run the COCO vehicle detector and return a list of padded bounding boxes
    for cars, motorcycles, buses and trucks.  Falls back to the full image
    if the vehicle model is unavailable or nothing is found.
    Returns list of (x1, y1, x2, y2) in absolute pixel coords.
    """
    if _vehicle_model is None:
        return [(0, 0, image.shape[1], image.shape[0])]

    results = _vehicle_model(image.copy(), conf=conf_thresh, verbose=False)
    rois = []
    h, w = image.shape[:2]
    for r in results:
        for box, cls_id in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        ):
            if cls_id not in _VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.astype(int)
            # Add padding around the vehicle so the plate near the bumper isn't clipped
            pw = int((x2 - x1) * pad_frac)
            ph = int((y2 - y1) * pad_frac)
            rois.append((
                max(0, x1 - pw), max(0, y1 - ph),
                min(w, x2 + pw), min(h, y2 + ph),
            ))

    if not rois:
        # No vehicles found — fall back to full image so we never miss a plate
        rois = [(0, 0, w, h)]
    return rois


def _run_yolo(image, conf_thresh, iou_thresh, imgsz=None):
    """Run the plate detection model on a (possibly cropped) image region."""
    kwargs = dict(conf=conf_thresh, iou=iou_thresh, verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz
    results = model(image.copy(), **kwargs)
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        order = confs.argsort()[::-1]
        for box, conf in zip(boxes[order], confs[order]):
            detections.append((box, float(conf)))
    return detections


def detect_and_read(image, conf_thresh=0.25, iou_thresh=0.4):
    """
    Two-stage detection:
      1. Locate vehicles (car / motorcycle / bus / truck) in the full image.
      2. Run the plate detector inside each vehicle ROI.
    Falls back to full-image plate detection if no vehicles are found.
    A second lower-confidence pass is attempted when the first finds nothing.
    Duplicate plates (same text) across overlapping ROIs are de-duplicated.
    """
    rois = _get_vehicle_rois(image)
    final_texts = []
    seen_texts = set()

    for (rx1, ry1, rx2, ry2) in rois:
        roi_img = image[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            continue

        detections = _run_yolo(roi_img, conf_thresh, iou_thresh)
        if not detections:
            detections = _run_yolo(roi_img, conf_thresh=0.15, iou_thresh=iou_thresh, imgsz=1280)

        for (x1, y1, x2, y2), conf in detections:
            # Translate box coords back to full-image space for the crop
            abs_x1, abs_y1 = rx1 + int(x1), ry1 + int(y1)
            abs_x2, abs_y2 = rx1 + int(x2), ry1 + int(y2)
            plate_crop = pad_crop(image, abs_x1, abs_y1, abs_x2, abs_y2)
            if plate_crop.size == 0:
                continue
            text = read_plate_text(plate_crop)
            # Skip empty or duplicate plate texts
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            final_texts.append((plate_crop, text, conf))

    return final_texts


# ---------------------------------------------------------
# Helper: lookup plate and optionally send email
# ---------------------------------------------------------
def handle_plate_lookup(plate_text, smtp_user, smtp_password, auto_email, extra_violations=None):
    """Look up plate_text in DB, check violations, and optionally send challan email."""
    if not plate_text:
        return

    extra_violations = extra_violations or []
    # Session state keys scoped to this plate
    status_key = "email_status_{}".format(plate_text)

    owner = lookup_plate(plate_text)

    if owner:
        # Check document violations and merge with any runtime violations (e.g. helmet)
        violations = check_document_violations(owner) + extra_violations

        st.success(
            "**Owner Found!**\n\n"
            "- **Name:** {}\n"
            "- **Plate:** {}\n"
            "- **Contact:** {}\n"
            "- **Email:** {}".format(
                owner['name'], owner['plate_number'],
                owner['contact_number'], owner.get('email', 'N/A')
            )
        )

        # Document status
        if violations:
            st.error(
                "⚠️ **Violations Detected:**\n" +
                "\n".join(f"- {v}" for v in violations)
            )
        else:
            st.info("✅ All documents are valid.")

        body = build_alert_message(owner['name'], owner['plate_number'], violations)
        subject = build_alert_subject(owner['plate_number'], violations)
        recipient = owner.get('email', '').strip()

        if not recipient:
            st.warning("No email address on record for this owner.")
            return

        # Only offer/send email when there are actual violations
        if not violations:
            return

        if auto_email and smtp_user and smtp_password:
            # Only send once per detection (avoid re-sending on every rerun)
            if status_key not in st.session_state:
                ok, detail = send_email(smtp_user, smtp_password, recipient, subject, body)
                st.session_state[status_key] = ("ok" if ok else "fail", detail)

            result, detail = st.session_state[status_key]
            if result == "ok":
                st.success("✅ Challan email sent automatically to **{}**.".format(recipient))
            else:
                st.error("❌ Auto-email failed: {}".format(detail))
        else:
            label = "📧 Send Challan Email" if violations else "📧 Send Alert Email"
            with st.expander(label):
                custom_body = st.text_area(
                    "Message", value=body, key="msg_{}".format(plate_text)
                )
                if st.button("Send Email", key="send_{}".format(plate_text)):
                    if not smtp_user or not smtp_password:
                        st.session_state[status_key] = ("fail", "Gmail credentials not configured. Go to the Settings tab.")
                    else:
                        ok, detail = send_email(
                            smtp_user, smtp_password, recipient, subject, custom_body
                        )
                        st.session_state[status_key] = ("ok" if ok else "fail", detail)

                # Always show the last send result if available
                if status_key in st.session_state:
                    result, detail = st.session_state[status_key]
                    if result == "ok":
                        st.success("✅ Email sent to **{}**.".format(recipient))
                    else:
                        st.error("❌ Failed: {}".format(detail))
    else:
        st.warning("Plate **{}** is **not registered** in the database.".format(plate_text))
        if extra_violations:
            st.error(
                "⚠️ **Violations Detected:**\n" +
                "\n".join("- {}".format(v) for v in extra_violations)
            )


# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="ALPR System", page_icon="🚗", layout="wide")
st.title("🚗 Automatic License Plate Detection & Recognition (ALPR)")

# Persist Gmail credentials across reruns via session_state
if "smtp_user" not in st.session_state:
    st.session_state["smtp_user"] = os.environ.get("GMAIL_ADDRESS", "")
if "smtp_password" not in st.session_state:
    st.session_state["smtp_password"] = os.environ.get("GMAIL_APP_PASSWORD", "")
if "auto_email" not in st.session_state:
    st.session_state["auto_email"] = False

tab_detect, tab_db, tab_settings = st.tabs(
    ["🔍 Detection", "🗄️ Database", "⚙️ Settings"]
)

# ==========================================================
# TAB 1 — Detection
# ==========================================================
with tab_detect:
    smtp_user = st.session_state["smtp_user"]
    smtp_password = st.session_state["smtp_password"]
    auto_email = st.session_state["auto_email"]

    left_col, right_col = st.columns([1.2, 1])

    # Persist detection outputs across reruns so manual Send Email button works
    if "last_outputs" not in st.session_state:
        st.session_state["last_outputs"] = []   # list of (crop_bytes, text, conf)

    # ---------- IMAGE ----------
    with left_col:
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"]
        )
        run_btn = st.button("🔍 Run Detection")

    for _key, _default in [
        ("helmet_violation", False),
        ("helmet_count", 0),
        ("person_detected", False),
        ("detection_ran", False),
    ]:
        if _key not in st.session_state:
            st.session_state[_key] = _default

    if uploaded_file is not None:
        # Clear previous results whenever a different file is uploaded
        file_key = "{}_{}".format(uploaded_file.name, uploaded_file.size)
        if st.session_state.get("last_file_key") != file_key:
            st.session_state["last_outputs"] = []
            st.session_state["helmet_violation"] = False
            st.session_state["helmet_count"] = 0
            st.session_state["person_detected"] = False
            st.session_state["detection_ran"] = False
            for _k in [k for k in st.session_state if k.startswith("email_status_")]:
                del st.session_state[_k]
            st.session_state["last_file_key"] = file_key

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()

        # Decode directly from bytes — no lossy JPEG re-encode via tempfile
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to read the uploaded image. Please try a different file.")
            st.stop()

        with left_col:
            # Display original bytes so browser renders at full uploaded quality
            st.image(file_bytes, caption="Uploaded Image", use_column_width=True)

        if run_btn:
            with st.spinner("Running detection pipeline..."):
                raw_outputs = detect_and_read(image)
                annotated_img, person_detected, helmet_violation, no_helmet_count = detect_helmet_violations(image)

            st.session_state["person_detected"] = person_detected
            st.session_state["helmet_violation"] = helmet_violation
            st.session_state["helmet_count"] = no_helmet_count
            st.session_state["detection_ran"] = True

            # Show helmet-annotated image in left column
            with left_col:
                if _helmet_model is not None:
                    caption = (
                        "🟢 Person detected — helmet compliant" if (person_detected and not helmet_violation)
                        else "🔴 Person detected — NO HELMET" if (person_detected and helmet_violation)
                        else "🚗 No rider detected"
                    )
                    st.image(
                        cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        caption=caption,
                        use_column_width=True,
                    )

            # Encode crops as lossless PNG bytes so they survive reruns
            stored = []
            for crop, text, conf in raw_outputs:
                ok_enc, buf = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                stored.append((buf.tobytes() if ok_enc else None, text, conf))
            st.session_state["last_outputs"] = stored
            for key in list(st.session_state.keys()):
                if key.startswith("email_status_"):
                    del st.session_state[key]

        # Always render the right column once detection has run
        if st.session_state.get("detection_ran"):
            person_det = st.session_state.get("person_detected", False)
            helm_viol  = st.session_state.get("helmet_violation", False)
            helm_cnt   = st.session_state.get("helmet_count", 0)
            _extra_violations = [HELMET_VIOLATION_MSG] if person_det and helm_viol else []

            with right_col:
                st.subheader("📌 Detection Results")

                # ── Helmet status ──────────────────────────────────
                if person_det:
                    if helm_viol:
                        st.error(
                            "🪖 **HELMET VIOLATION** — {} rider(s) detected without a helmet.".format(helm_cnt)
                        )
                    else:
                        st.success("✅ Rider detected — helmet worn correctly.")
                else:
                    st.info("🚗 No rider detected — document check only.")

                st.divider()

                # ── Plate details ──────────────────────────────────
                outputs = st.session_state["last_outputs"]
                if outputs:
                    st.markdown("**📌 Detected Plates & Challan**")
                    for idx, (crop_bytes, text, conf) in enumerate(outputs):
                        if crop_bytes:
                            st.image(crop_bytes, caption="Plate #{} (conf: {:.2f})".format(idx + 1, conf), width=300)
                        st.info("Plate Text: **{}**".format(text))
                        handle_plate_lookup(
                            text, smtp_user, smtp_password, auto_email,
                            extra_violations=_extra_violations,
                        )
                        st.divider()
                else:
                    st.warning("⚠️ No license plates detected in this image.")



# ==========================================================
# TAB 2 — Database Management
# ==========================================================
with tab_db:
    st.subheader("🗄️ Vehicle Owner Database")

    # Initialise edit state
    if "edit_plate" not in st.session_state:
        st.session_state["edit_plate"] = None

    # -- Add / Edit record form --
    is_editing = st.session_state["edit_plate"] is not None
    form_title = "✏️ Edit Record: {}".format(st.session_state["edit_plate"]) if is_editing else "➕ Add New Record"

    # Pre-fill values when editing
    if is_editing:
        _prefill = lookup_plate(st.session_state["edit_plate"]) or {}
    else:
        _prefill = {}

    with st.expander(form_title, expanded=True):
        with st.form("record_form"):
            col1, col2 = st.columns(2)
            with col1:
                form_plate = st.text_input(
                    "Plate Number",
                    value=_prefill.get("plate_number", ""),
                    placeholder="MH14EH5819",
                    disabled=is_editing,   # plate is the PK — don't allow changing it
                )
                form_name = st.text_input(
                    "Owner Name",
                    value=_prefill.get("name", ""),
                    placeholder="Rahul Sharma",
                )
            with col2:
                form_contact = st.text_input(
                    "Contact Number (10 digits)",
                    value=_prefill.get("contact_number", ""),
                    placeholder="9876543210",
                )
                form_email = st.text_input(
                    "Email Address",
                    value=_prefill.get("email", ""),
                    placeholder="rahul@example.com",
                )

            st.markdown("**Document Expiry Dates** *(leave unchanged if not applicable)*")
            dcol1, dcol2, dcol3 = st.columns(3)

            def _parse_date(val):
                """Return datetime.date from 'YYYY-MM-DD' string, or today if blank."""
                try:
                    return date.fromisoformat(val) if val else date.today()
                except ValueError:
                    return date.today()

            with dcol1:
                form_rc = st.date_input(
                    "RC Expiry",
                    value=_parse_date(_prefill.get("rc_expiry", "")),
                    key="rc_date",
                )
            with dcol2:
                form_puc = st.date_input(
                    "PUC Expiry",
                    value=_parse_date(_prefill.get("puc_expiry", "")),
                    key="puc_date",
                )
            with dcol3:
                form_ins = st.date_input(
                    "Insurance Expiry",
                    value=_parse_date(_prefill.get("insurance_expiry", "")),
                    key="ins_date",
                )

            save_btn = st.form_submit_button("💾 Save Changes" if is_editing else "➕ Add Record")

        if save_btn:
            plate_to_save = st.session_state["edit_plate"] if is_editing else form_plate
            ok, msg = add_owner(
                plate_to_save, form_name, form_contact, form_email,
                rc_expiry=form_rc.isoformat(),
                puc_expiry=form_puc.isoformat(),
                insurance_expiry=form_ins.isoformat(),
            )
            if ok:
                st.success(msg)
                st.session_state["edit_plate"] = None
                st.rerun()
            else:
                st.error(msg)

        if is_editing and st.button("✖ Cancel Edit"):
            st.session_state["edit_plate"] = None
            st.rerun()

    st.divider()

    # -- View all records --
    st.subheader("📋 All Records")
    all_records = get_all_owners()

    if not all_records:
        st.info("No records in the database yet. Add one above.")
    else:
        df = pd.DataFrame(all_records)[[
            "plate_number", "name", "contact_number", "email",
            "rc_expiry", "puc_expiry", "insurance_expiry"
        ]]
        df.columns = ["Plate Number", "Name", "Contact", "Email",
                      "RC Expiry", "PUC Expiry", "Insurance Expiry"]

        today_str = date.today().isoformat()

        def _style_expiry(val):
            if val and val < today_str:
                return "background-color: #ffe0e0; color: #c00"
            return ""

        styled = df.style.applymap(
            _style_expiry, subset=["RC Expiry", "PUC Expiry", "Insurance Expiry"]
        )
        st.dataframe(styled, use_container_width=True)

        st.markdown("**Edit or delete a record:**")
        for record in all_records:
            col_info, col_edit, col_del = st.columns([4, 1, 1])
            with col_info:
                st.write(
                    "**{}** — {} | {} | {}".format(
                        record["plate_number"],
                        record["name"],
                        record["contact_number"],
                        record["email"] or "no email",
                    )
                )
            with col_edit:
                if st.button("✏️ Edit", key="edit_{}".format(record["plate_number"])):
                    st.session_state["edit_plate"] = record["plate_number"]
                    st.rerun()
            with col_del:
                if st.button("🗑️ Delete", key="del_{}".format(record["plate_number"])):
                    ok, msg = delete_owner(record["plate_number"])
                    if ok:
                        st.success(msg)
                        if st.session_state.get("edit_plate") == record["plate_number"]:
                            st.session_state["edit_plate"] = None
                        st.rerun()
                    else:
                        st.error(msg)


# ==========================================================
# TAB 3 — Settings
# ==========================================================
with tab_settings:
    st.subheader("⚙️ Settings")

    st.markdown(
        "Configure your **Gmail** credentials to send email notifications. "
        "You must use a **Gmail App Password**, not your regular Gmail password.\n\n"
        "**Setup steps:**\n"
        "1. Enable 2-Step Verification on your Google account.\n"
        "2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).\n"
        "3. Create an App Password → select **Mail** → copy the 16-character password."
    )

    smtp_user_input = st.text_input(
        "Gmail Address",
        value=st.session_state["smtp_user"],
        placeholder="yourname@gmail.com",
    )

    smtp_pass_input = st.text_input(
        "Gmail App Password",
        value=st.session_state["smtp_password"],
        type="password",
        placeholder="16-character App Password",
    )

    auto_email_input = st.toggle(
        "Auto-send email on plate detection",
        value=st.session_state["auto_email"],
        help="When enabled, an email is sent automatically every time a registered plate is detected.",
    )

    if st.button("Save Settings"):
        st.session_state["smtp_user"] = smtp_user_input
        st.session_state["smtp_password"] = smtp_pass_input
        st.session_state["auto_email"] = auto_email_input
        st.success("Settings saved for this session.")

    st.divider()

    st.markdown(
        "**Tip:** To persist credentials across sessions, set environment variables "
        "before launching the app:\n"
        "```bash\n"
        "export GMAIL_ADDRESS='yourname@gmail.com'\n"
        "export GMAIL_APP_PASSWORD='your_app_password'\n"
        "streamlit run models/app.py\n"
        "```"
    )
