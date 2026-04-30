# 📊 Demand Prediction Engine (Image-Based)

## 👤 Candidate Details

**Full Name:** Shreya Sidabache


---

## 🚀 Project Overview

This project builds a machine learning system to **predict product demand (quantity sold)** based on:

* Product design images
* Historical sales data

The system extracts visual features from images and combines them with pricing data to estimate expected demand.

---

## 🎯 Objective

To explore whether **product design (visual appearance)** can help predict how well a product will sell.

---

## 🧠 Approach Summary

1. **Data Processing**

   * Aggregated transactional data per product:

     * Total demand → `qty (sum)`
     * Average price → `rate (mean)`

2. **Challenge Identified**

   * No direct mapping between product codes and image filenames

3. **Solution Applied**

   * Used **sequential mapping** between images and product entries to build a working pipeline

4. **Feature Extraction**

   * Used **ResNet50 (pretrained)** to extract 2048-dim image features

5. **Model**

   * Combined image features + price
   * Trained **XGBoost Regressor**
   * Applied **log transformation** for stability

6. **Deployment**

   * Built a **Streamlit UI** for real-time prediction

---

## ⚙️ Tech Stack

* Python
* PyTorch (ResNet50)
* XGBoost
* Scikit-learn
* Pandas / NumPy
* Streamlit

---

## 🖥️ Working UI

The application allows users to:

* Upload a design image
* Get predicted demand (units)
* View demand category:

  * Low
  * Medium
  * High

---

## 📸 Sample Output

**Example Prediction:**

* Predicted Demand: ~23 units
* Category: Medium Demand

*(Add screenshot here if uploading to GitHub)*

---

## 📊 Model Performance

* MAE: ~14–15 units
* Relative Error: ~50–70%

👉 Performance is limited due to lack of direct image-product mapping.

---

## ⚠️ Limitations

* No true mapping between images and product codes
* Model learns general demand trends, not exact design impact
* Missing important features:

  * Product category
  * Seasonality
  * Trends

---

## 💡 Future Improvements

* Create proper **image ↔ product mapping**
* Use **CLIP embeddings** for better visual understanding
* Add metadata (category, color, style)
* Include time-series features
* Improve model with multimodal deep learning

---

## ▶️ How to Run

### 1. Install dependencies

```id="c3xz7n"
pip install -r requirements.txt
```

### 2. Train model

```id="ytx2g3"
python -m src.train
```

### 3. Run application

```id="r2w3rm"
streamlit run app.py
```

---

## 📁 Project Structure

```id="b8u2xk"
project/
│── Data/
│── Images/
│── models/
│── src/
│   ├── feature_extractor.py
│   ├── train.py
│   ├── predict.py
│── app.py
│── requirements.txt
│── README.md
```

---

## 🧠 Key Insight

> The main challenge was not model selection but **data alignment**.
> Without proper mapping between images and products, the model cannot learn true visual-demand relationships.

---

## ✅ Conclusion

This project successfully demonstrates:

* End-to-end ML pipeline
* Image-based feature extraction
* Demand prediction system
* Handling real-world data challenges

While predictions are approximate, the system shows how visual features can be integrated into demand forecasting.

---
