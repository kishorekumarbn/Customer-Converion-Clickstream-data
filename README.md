# 🛒 Customer Conversion Analysis for Online Shopping Using Clickstream Data

## 📌 Project Overview
This project leverages **Machine Learning** to analyze customer behavior using **Clickstream Data** and predict key insights. It includes:

✔️ **Regression** - Predicts product price based on user behavior.
✔️ **Classification** - Categorizes products into different price segments.
✔️ **Clustering** - Segments customers based on interaction patterns.
✔️ **Streamlit Web App** - Provides an interactive interface for predictions.

## 🚀 Features
- **📂 CSV Upload & Manual Input:** Choose between uploading a dataset or entering data manually.
- **🧠 Machine Learning Models:
- ** Uses **XGBoost Regressor,
   Linear Regression,
   Ridge,
   Lasso,
   Logistic Regression,
   Decision Tree Classifier,
   Random Forest Classifier
   DBSCAN
   and KMeans Clustering**.
- **🔍 Automated Data Preprocessing:** Handles missing values, encodes categorical features, and scales data.
- **📊 Intuitive Predictions:** Get real-time insights into pricing, classification, and customer segments.

## 🏗️ Tech Stack
- **Python** 🐍
- **Pandas & NumPy** 📊
- **Scikit-Learn & Multiple ML Algo** 🤖
- **Streamlit** 🎨
- **Pickle** 🏗️

## 🛠️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-conversion-analysis.git
   cd customer-conversion-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📥 Input Format
### **CSV Upload:**
- Ensure your dataset includes relevant columns: `page1_main_category`, `page2_clothing_model`, `colour`, `location`, `model_photography`, `page`, etc.

### **Manual Input Fields:**
- **Dropdowns & Selectboxes** for categorical inputs (e.g., `page1_main_category`, `location`).
- **Text input fields** for numerical values (e.g., `colour`).
- **Price & Price_2** included **only for clustering**.

## 📌 Usage Guide
1. **Select a task**: Regression, Classification, or Clustering.
2. **Choose input method**: Upload CSV or enter data manually.
3. **Get instant predictions**!

## 📊 Example Predictions
| Task          | Example Output |
|--------------|---------------|
| Regression   | Predicted Price: $29.99 |
| Classification | Predicted Price Category: 1 |
| Clustering   | Cluster: 2 |

## 👨‍💻 Contributing
Feel free to fork this repo and contribute! 🚀


By:
Kishore Kumar B N
gmail: kishorekumarbn18@gmail.com
