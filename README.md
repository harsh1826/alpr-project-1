# 🚗 Indian ALPR (YOLOv8 + EasyOCR)

A fast, lightweight Automated License Plate Recognition system built for Indian vehicle formats. 

- **Detection:** YOLOv8 (`best.pt`)
- **Extraction:** EasyOCR with 3x upscaling & CLAHE contrast enhancement
- **Validation:** Strict Indian RTO regex parsing (removes "IND" artifacts)
- **Deployment:** Streamlit

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
