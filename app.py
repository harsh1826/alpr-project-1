import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import time
import re

# 1. Page Configuration
st.set_page_config(page_title="Project 1: Indian ALPR", layout="wide")
st.title("Project 1: Indian License Plate Detector")
st.caption("YOLOv8 + EasyOCR | RTO Format Enforced | IND Filtering (High Contrast)")

# 2. Model Initialization
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

model, reader = load_models()

# 3. All Official Indian State & Union Territory Codes
VALID_INDIAN_STATES = {
    # States
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TG", "TS", "TR", "UP", "UK", "WB",
    # Union Territories
    "AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY"
}

# 4. Stroke-Preserving Enhancement (Updated for Q/O and 0/6 fixes)
def enhance_plate(crop):
    # Add border padding so edge characters are not clipped
    padded = cv2.copyMakeBorder(crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Increased to 3x scaling so Q-tails and 0-loops have more distinct pixel definition
    resized = cv2.resize(padded, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter for noise reduction without blurring edges
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # Aggressive CLAHE (clipLimit 3.0) to force dark text to stand out against shadows
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)

# 5. Strict Indian Plate Parser & Sanitizer
def sanitize_indian_plate(raw_text):
    # Uppercase and strip spaces/punctuation
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    # Remove HSRP "IND" tag anywhere it appears
    cleaned = re.sub(r'IND', '', cleaned)
    
    if not cleaned:
        return "Unreadable"

    # Positional character correction mappings
    char_to_num = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

    # Step A: Find the starting point of the state code
    start_idx = -1
    for i in range(len(cleaned) - 1):
        if cleaned[i:i+2] in VALID_INDIAN_STATES:
            start_idx = i
            break
            
    # If no exact state code was matched, check if digits replaced letters at the start
    if start_idx == -1 and len(cleaned) >= 2:
        candidate_state = num_to_char.get(cleaned[0], cleaned[0]) + num_to_char.get(cleaned[1], cleaned[1])
        if candidate_state in VALID_INDIAN_STATES:
            cleaned = candidate_state + cleaned[2:]
            start_idx = 0

    candidate = cleaned[start_idx:] if start_idx != -1 else cleaned

    # Step B: Match Standard 10-Character Structure (LL DD LL DDDD)
    # Truncates trailing artifacts like the '1' from '1IND'
    if len(candidate) >= 10:
        c = list(candidate[:10])
        state = num_to_char.get(c[0], c[0]) + num_to_char.get(c[1], c[1])
        d0, d1 = char_to_num.get(c[2], c[2]), char_to_num.get(c[3], c[3])
        l0, l1 = num_to_char.get(c[4], c[4]), num_to_char.get(c[5], c[5])
        n = [char_to_num.get(ch, ch) for ch in c[6:10]]

        if (state in VALID_INDIAN_STATES and d0.isdigit() and d1.isdigit() and
            l0.isalpha() and l1.isalpha() and all(digit.isdigit() for digit in n)):
            return f"{state}{d0}{d1}{l0}{l1}{''.join(n)}"

    # Step C: Match 9-Character Variant 1 (LL DD L DDDD)
    if len(candidate) >= 9:
        c = list(candidate[:9])
        state = num_to_char.get(c[0], c[0]) + num_to_char.get(c[1], c[1])
        d0, d1 = char_to_num.get(c[2], c[2]), char_to_num.get(c[3], c[3])
        l0 = num_to_char.get(c[4], c[4])
        n = [char_to_num.get(ch, ch) for ch in c[5:9]]

        if (state in VALID_INDIAN_STATES and d0.isdigit() and d1.isdigit() and
            l0.isalpha() and all(digit.isdigit() for digit in n)):
            return f"{state}{d0}{d1}{l0}{''.join(n)}"

    # Step D: Match 9-Character Variant 2 (LL D LL DDDD) e.g., DL 3 CC 1234
    if len(candidate) >= 9:
        c = list(candidate[:9])
        state = num_to_char.get(c[0], c[0]) + num_to_char.get(c[1], c[1])
        d0 = char_to_num.get(c[2], c[2])
        l0, l1 = num_to_char.get(c[3], c[3]), num_to_char.get(c[4], c[4])
        n = [char_to_num.get(ch, ch) for ch in c[5:9]]

        if (state in VALID_INDIAN_STATES and d0.isdigit() and
            l0.isalpha() and l1.isalpha() and all(digit.isdigit() for digit in n)):
            return f"{state}{d0}{l0}{l1}{''.join(n)}"

    # Fallback if structure is non-standard
    return candidate[:10] if candidate else "Unreadable"

# 6. Streamlit Execution Pipeline
uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Original Image", use_container_width=True)
    
    if st.button("Detect Plate"):
        start_time = time.time()
        
        with st.spinner("Processing..."):
            results = model.predict(img_array, conf=0.25)[0]
            boxes = results.boxes.data.tolist()
            
            if not boxes:
                st.error("No license plate detected. Try another image.")
            else:
                for box in boxes:
                    x1, y1, x2, y2, score, class_id = box
                    
                    crop = img_array[int(y1):int(y2), int(x1):int(x2)]
                    enhanced_crop = enhance_plate(crop)
                    
                    # Sub-second OCR extraction (greedy mode) updated for contrast/scaling
                    text_results = reader.readtext(
                        enhanced_crop,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        mag_ratio=2.0,       # Forces EasyOCR to double the resolution internally
                        adjust_contrast=0.5  # Re-evaluates lighting to separate dirt from loops
                    )
                    
                    if text_results:
                        # Exclude any isolated block that is just "IND"
                        valid_segments = [res[1] for res in text_results if res[1].strip() != "IND"]
                        raw_extracted = "".join(valid_segments)
                        formatted_plate = sanitize_indian_plate(raw_extracted)
                    else:
                        formatted_plate = "Unreadable"
                    
                    elapsed_time = time.time() - start_time
                    
                    st.divider()
                    st.subheader("Detection Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(enhanced_crop, caption=f"Padded Enhanced Crop (Conf: {score:.2f})", use_container_width=True)
                    with col2:
                        st.success("Extraction Complete")
                        st.metric(label="Detected Number Plate", value=formatted_plate)
                        st.metric(label="Time Taken", value=f"{elapsed_time:.2f} seconds")