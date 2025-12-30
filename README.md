# 🌴 SawitScope • Ripeness Classifier

Dashboard analitik dan prediksi kematangan tandan sawit
(Mentah • Matang • Busuk)

## 📌 Fitur
- Image classification (Single / Batch)
- Confidence & ambiguity analysis
- Multi-model: XGBoost, EfficientNet, MaxViT
- Streamlit dashboard

## 📦 Model Files
⚠️ File model tidak disertakan di repository karena ukuran besar.

Silakan unduh model dan letakkan di folder:
`sawit_models/`

## ▶️ Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run src/app.py
