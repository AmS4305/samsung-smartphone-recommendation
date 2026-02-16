# 📱 Samsung SmartPick — Smartphone Recommendation System

An AI-powered Samsung smartphone **price predictor** and **phone recommendation engine** built with Machine Learning and Streamlit.  
Trained on real Samsung smartphone data scraped from Flipkart, featuring KMeans clustering, GradientBoosting regression, and cosine-similarity-based recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🖼️ App Preview

<!-- 📸 INSTRUCTIONS: Take a screenshot of the running app and save it as screenshots/app_preview.png -->
<!-- Then uncomment the line below: -->
<!-- ![Samsung SmartPick Preview](screenshots/app_preview.png) -->

> **To add a preview image:** Run the app, take a screenshot, save it to `screenshots/app_preview.png`, and uncomment the image line above.

---

## ✨ Features

### 💰 Price Predictor

- Configure specs like **RAM, Storage, Camera, Display Size, Network**, and **Android version**
- Predicts the estimated market price using a **GradientBoosting Regressor**
- Shows model performance metrics (R², Adjusted R², MAE)

### 🔍 Phone Recommender

- Select any Samsung phone from the database
- Finds the **most similar phones** using **cosine similarity** on spec features
- Displays phone images, similarity scores with visual bars, and detailed spec tags

### 📊 Data Pipeline (Jupyter Notebook)

- Data cleaning & feature extraction from raw Flipkart scraped data
- KNN imputation for missing values
- One-hot encoding for categorical features (OS, network type)
- **KMeans clustering** (K=3) to segment phones into Flagship, Mid-Range, and Budget
- Multiple regression models compared (Linear, DecisionTree, RandomForest, SVR, GradientBoosting)

---

## 🛠️ Tech Stack

| Component             | Technology                                     |
| --------------------- | ---------------------------------------------- |
| **Frontend**          | Streamlit with custom CSS (Samsung dark theme) |
| **ML Models**         | scikit-learn (GradientBoosting, KMeans, KNN)   |
| **Similarity Engine** | Cosine Similarity (StandardScaler + pairwise)  |
| **Data Processing**   | Pandas, NumPy                                  |
| **Visualization**     | Matplotlib, Seaborn (in notebook)              |
| **Data Source**       | Flipkart Samsung phone listings                |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/samsung-smartphone-recommendation.git
cd samsung-smartphone-recommendation

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## 📂 Project Structure

```
samsung-smartphone-recommendation/
├── app.py                                          # 🎨 Streamlit web app
├── samsung-smartphone-recommendation-system.ipynb  # 📓 Full ML pipeline notebook
├── project1.csv                                    # 📊 Dataset 1 (cleaned specs)
├── project2.csv                                    # 📊 Dataset 2 (raw Flipkart data)
├── requirements.txt                                # 📦 Python dependencies
├── screenshots/                                    # 🖼️ App preview images
│   └── app_preview.png
├── .gitignore
└── README.md
```

---

## 📓 Notebook Walkthrough

The Jupyter notebook (`samsung-smartphone-recommendation-system.ipynb`) contains the full ML pipeline:

1. **Data Loading** — Two CSV datasets with Samsung phone specs from Flipkart
2. **Data Cleaning** — Regex extraction of specs from raw text fields (storage, camera, display, etc.)
3. **Missing Value Imputation** — Random sampling for categorical, KNN imputation for numerical
4. **Feature Engineering** — One-hot encoding of OS and network type
5. **EDA** — Distribution plots, price vs. rating scatter plots
6. **Clustering** — KMeans (K=3) with elbow method visualization + PCA 2D plot
7. **Model Training** — 5 regression models compared; GradientBoosting selected as best
8. **Feature Importance** — DecisionTree-based feature ranking
9. **Interactive Widgets** — Price predictor and phone recommender (ipywidgets)

---

## 📊 Model Performance

| Model                         | R² Score     |
| ----------------------------- | ------------ |
| LinearRegression              | ~0.72        |
| DecisionTreeRegressor         | ~0.87        |
| RandomForestRegressor         | ~0.91        |
| SVR                           | ~0.15        |
| **GradientBoostingRegressor** | **~0.94** ✅ |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or feature requests
- Submit pull requests
- Suggest UI improvements

---

## 📄 License

This project is provided as-is for educational and personal use.

---

<p align="center">
  <b>Built with ❤️ using Python, scikit-learn, and Streamlit</b>
</p>
