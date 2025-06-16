# Student Depression Risk Prediction System

A comprehensive machine learning application that predicts depression risk in students using XGBoost algorithm with 85.2% accuracy. The system provides both Turkish and English interfaces for analyzing student mental health data.

## 🚀 Features

- **Bilingual Support**: Available in both Turkish and English
- **Interactive Dashboard**: Multi-page application with comprehensive analysis
- **Data Exploration**: Visualize dataset distributions and statistical insights
- **Model Performance**: Display accuracy metrics and feature importance analysis
- **Risk Prediction**: Interactive form to predict depression risk for individual students
- **Educational Content**: Step-by-step explanation of the ML development process
- **Professional Recommendations**: Evidence-based suggestions for intervention

## 📁 Project Structure

```
ML-Student-Depression/
├── streamlit_app.py              # English Streamlit application
├── streamlit_tr.py               # Turkish Streamlit application (Türkçe)
├── requirements.txt              # Python dependencies
├── main.ipynb                    # Original model development notebook
├── no_suicidal.ipynb            # Alternative analysis notebook
├── Student Depression Dataset.csv # Training dataset
├── xgboost_depression_model.pkl  # Trained XGBoost model
├── xgboost_scaler.pkl           # Feature scaler for model
├── scaler.pkl                   # Additional scaler file
├── feature_names.pkl            # Model feature names
└── README.md                    # Project documentation
```

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML-Student-Depression
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   
   **For English version:**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   **For Turkish version (Türkçe):**
   ```bash
   streamlit run streamlit_tr.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

## 📊 Application Sections

Both applications (English and Turkish) contain the same comprehensive sections:

### 🏠 About Project / Proje Hakkında
- Project overview and objectives
- Model accuracy and dataset statistics
- Key features and system capabilities

### � How I Did It? / Nasıl Yaptım?
- Step-by-step development methodology
- Technical explanations made accessible
- Technology stack and tools used
- Model selection and optimization process

### 📊 Data Analysis / Veri Analizi
- Dataset overview and statistics
- Interactive data visualizations
- Depression distribution analysis
- Key statistical findings

### 🤖 Model Performance / Model Performansı
- Detailed performance metrics (Accuracy, Precision, Recall, F1-Score)
- Feature importance visualization
- Model comparison analysis
- Algorithm explanations and working principles

### 🔮 Risk Prediction / Risk Tahmini
- Interactive form for student data input
- Real-time depression risk assessment
- Risk level classification with recommendations
- Visual probability gauge and detailed explanations

### 📈 Findings and Recommendations / Bulgular ve Öneriler
- Key research findings and insights
- Evidence-based recommendations
- Institutional and individual guidance
- Future development possibilities

## 🎯 Model Information

- **Algorithm**: XGBoost Classifier (Extreme Gradient Boosting)
- **Training Accuracy**: 85.2%
- **Dataset Size**: 27,000+ student records
- **Features**: 16 engineered features including psychological, academic, and lifestyle indicators
- **Risk Categories**: Low 🟢, Medium 🟡, High 🟠, Very High 🔴
- **Model Files**: Pre-trained model, scaler, and feature mappings included

## 🔍 Key Predictive Features

The model analyzes multiple factors to assess depression risk:

1. **Psychological Indicators**
   - Suicidal thoughts history (highest impact factor)
   - Family mental health history
   - Overall life satisfaction

2. **Stress Factors**
   - Financial stress levels
   - Academic pressure
   - Work-life balance

3. **Lifestyle Variables**
   - Sleep duration patterns
   - Dietary habits
   - Daily study/work hours

4. **Academic Factors**
   - Study satisfaction
   - Grade point average (CGPA)
   - Degree level

5. **Demographic Information**
   - Age group (18-25 identified as high-risk)
   - Gender
   - Education level

## 🔬 Technical Details

- **Development Environment**: Python 3.8+ with Jupyter Notebooks
- **ML Libraries**: Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Framework**: Streamlit for interactive applications
- **Model Persistence**: Joblib for model serialization

## 🎨 Customization

You can customize the applications by:

- **Styling**: Modify CSS styles in the `st.markdown()` sections for custom themes
- **Visualizations**: Add new charts or modify existing plot configurations
- **Risk Assessment**: Update risk thresholds and recommendation logic
- **Input Fields**: Add new prediction parameters or modify existing form fields
- **Language**: Extend to other languages following the bilingual pattern
- **Model Updates**: Replace model files to use updated or different algorithms

## 🌐 Bilingual Support

This project demonstrates international accessibility:

- **streamlit_app.py**: Complete English interface
- **streamlit_tr.py**: Full Turkish interface (Türkçe arayüz)
- Both versions maintain identical functionality
- Easy to extend to additional languages
- Cultural sensitivity in mental health terminology

## 🚨 Important Notes

- **File Dependencies**: Ensure all model files (`.pkl`) are in the same directory
- **Dataset**: The CSV file must be named exactly "Student Depression Dataset.csv"
- **Python Version**: Requires Python 3.8 or higher
- **Memory**: XGBoost model requires sufficient RAM for large predictions
- **Production**: Add comprehensive data validation and error handling for deployment

## 🤝 Usage Guidelines

### For Researchers and Developers:
- Use `main.ipynb` to understand the model development process
- Examine feature engineering techniques in the notebooks
- Adapt the methodology for similar mental health prediction tasks

### For Educators and Institutions:
- Deploy the application for educational demonstrations
- Use risk predictions as supplementary screening tools
- Implement recommendations for institutional mental health programs

### For Students and General Users:
- Navigate using the sidebar menu for different analysis sections
- Complete all fields in Risk Prediction for accurate results
- Remember that results are supplementary to professional assessment

## � Ethical Considerations & Disclaimer

**Important Medical Disclaimer:**
- This tool is designed for **educational and research purposes only**
- Results should **never replace professional mental health assessment**
- Always consult qualified mental health professionals for diagnosis and treatment
- The system provides risk indicators, not definitive medical diagnoses

**Ethical AI Use:**
- Model predictions should supplement, not replace, human judgment
- Consider potential biases in training data and model decisions
- Maintain privacy and confidentiality of student information
- Use predictions responsibly within appropriate institutional frameworks

**Data Privacy:**
- No user input data is stored or transmitted
- All processing occurs locally within the application
- Follow institutional guidelines for handling student mental health data

## 📞 Troubleshooting & Support

**Common Issues:**

1. **Missing Dependencies:**
   ```bash
   pip install --upgrade streamlit plotly scikit-learn xgboost pandas numpy joblib seaborn matplotlib
   ```

2. **Model File Errors:**
   - Verify all `.pkl` files are in the same directory as the Python scripts
   - Check file permissions and accessibility

3. **Dataset Issues:**
   - Ensure CSV file is named exactly "Student Depression Dataset.csv"
   - Verify file encoding (UTF-8 recommended)

4. **Performance Issues:**
   - Close other applications to free memory
   - Consider using a virtual environment
   - Update to the latest Python version (3.8+)

**Getting Help:**
- Check GitHub Issues for known problems
- Review Streamlit documentation for deployment issues
- Verify system requirements and compatibility

## 🚀 Future Enhancements

**Planned Features:**
- Real-time monitoring dashboard
- Integration with learning management systems
- Mobile-responsive design
- API endpoints for institutional integration
- Advanced visualization options
- Multi-language support expansion

**Research Opportunities:**
- Longitudinal tracking capabilities
- Integration with wearable device data
- Social media sentiment analysis
- Intervention effectiveness tracking
- Cross-cultural validation studies

---

**Project Status:** ✅ Complete and Ready for Use  
**Last Updated:** June 2025  
**License:** Educational Use  
**Developed for:** ML Student Depression Analysis Project
