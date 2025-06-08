# Student Depression Risk Predictor - Streamlit Presentation

This is a comprehensive Streamlit application for presenting your Machine Learning model that predicts depression risk in students.

## 🚀 Features

- **Interactive Dashboard**: Multi-page application with different sections
- **Data Exploration**: Visualize dataset distributions and correlations
- **Model Performance**: Display accuracy metrics and feature importance
- **Risk Prediction**: Interactive form to predict depression risk for new students
- **Insights & Analytics**: Key findings and recommendations

## 📁 Project Structure

```
ML-Student-Depression/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── main.ipynb                   # Original Jupyter notebook
├── Student Depression Dataset.csv # Dataset
├── xgboost_depression_model.pkl  # Trained XGBoost model
├── xgboost_scaler.pkl           # Data scaler
├── feature_names.pkl            # Feature names
└── README.md                    # This file
```

## 🔧 Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

## 📊 Application Sections

### 🏠 Overview
- Project summary and key metrics
- Model accuracy and dataset statistics
- Feature overview

### 📊 Data Exploration
- Dataset overview and statistics
- Interactive data visualizations
- Correlation heatmaps
- Distribution plots

### 🤖 Model Performance
- Accuracy metrics (Accuracy, Precision, Recall, F1-Score)
- Feature importance visualization
- Model comparison charts

### 🔮 Risk Prediction
- Interactive form for student data input
- Real-time depression risk prediction
- Risk level assessment with recommendations
- Visual risk probability gauge

### 📈 Insights & Analytics
- Key findings from the analysis
- Risk factor impact analysis
- Recommendations for institutions and students
- Future enhancement suggestions

## 🎯 Model Information

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 85.2%
- **Key Features**: 16 engineered features including stress indicators, academic factors, and lifestyle variables
- **Risk Levels**: Low, Medium, High, Very High

## 🔍 Key Predictive Features

1. Suicidal thoughts history
2. Financial stress levels
3. Sleep duration patterns
4. Academic pressure
5. Study satisfaction
6. Age group (18-25 high risk)
7. Dietary habits
8. Work-life balance

## 🎨 Customization

You can customize the application by:

- Modifying the CSS styles in the `st.markdown()` sections
- Adding new visualizations or metrics
- Updating the risk assessment logic
- Adding new input fields for prediction

## 🚨 Important Notes

- Ensure all model files (`.pkl`) are in the same directory as the Streamlit app
- The dataset file should be named exactly "Student Depression Dataset.csv"
- For production use, consider adding data validation and error handling

## 🤝 Usage Tips

- Use the sidebar navigation to switch between different sections
- In the Risk Prediction section, fill all fields for accurate predictions
- The model provides probability scores and risk levels with actionable recommendations
- Charts are interactive - hover for more details

## 🔒 Disclaimer

This tool is for educational and research purposes. It should not replace professional mental health assessment or treatment. Always consult qualified mental health professionals for serious concerns.

## 📞 Support

If you encounter any issues:
1. Check that all required files are present
2. Verify Python dependencies are installed
3. Ensure you're using a compatible Python version (3.8+)

---

**Developed for ML Student Depression Analysis Project**
