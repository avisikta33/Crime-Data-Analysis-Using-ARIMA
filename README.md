# Crime Data Analysis Using ARIMA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## 📋 Project Overview

A comprehensive statistical analysis project that applies **ARIMA (AutoRegressive Integrated Moving Average)** time series modeling to analyze and forecast crime patterns in Western & Middle India. This project leverages historical crime data (2001-2012) to identify trends, predict future crime rates, and provide actionable insights for law enforcement agencies.

### 🎓 Academic Details
- **Project Duration:** March 2023 - July 2023
- **Institution:** Asutosh College, University of Calcutta
- **Department:** Statistics
- **Mentor:** Prof. Shirshendu Mukherjee
- **Author:** Avisikta Das

---

## 🎯 Objectives

1. **Extract and analyze** historical crime patterns from 6 states across 6 crime categories
2. **Identify trends** and seasonality in crime data using statistical methods
3. **Preprocess and clean** datasets ensuring accuracy and consistency
4. **Apply ARIMA modeling** to forecast future crime rates (2013-2015)
5. **Deliver actionable insights** for proactive policing, resource allocation, and policy planning
6. **Calculate Crime Index** for safety assessment of each state

---

## 📊 Dataset

### Geographic Coverage
- **States Analyzed:** 
  - Chhattisgarh
  - Gujarat
  - Haryana
  - Madhya Pradesh
  - Maharashtra
  - Rajasthan

### Crime Categories
1. Murder
2. Rape
3. Kidnapping
4. Robbery
5. Riots
6. Counterfeiting

### Time Period
- **Years:** 2001 - 2012 (12 years)
- **Total Data Points:** 432 (6 states × 6 crimes × 12 years)

### Data Source
Crime data extracted from official records available on [Kaggle](https://www.kaggle.com/datasets/rajanand/crime-in-india)

---

## 🚀 Usage

### Basic Execution
Run the main analysis script:
```bash
python crime_arima_analysis.py
```

### Custom Analysis
To analyze specific state and crime type:
```python
from crime_arima_analysis import analyze_state

# Analyze Murder in Chhattisgarh
model, forecast = analyze_state('Chhattisgarh', 'MURDER')

# Analyze Kidnapping in Gujarat
model, forecast = analyze_state('Gujarat', 'KIDNAPPING')
```

### Generate Summary Report
```python
from crime_arima_analysis import generate_summary_report

report_df = generate_summary_report()
```

---

## 📈 Features

### 1. **Data Preprocessing**
- ✅ Data extraction from PDF documents
- ✅ Data cleaning and validation
- ✅ Missing value handling
- ✅ Time series formatting

### 2. **Descriptive Analysis**
- 📊 Mean, median, standard deviation
- 📈 Trend identification (increasing/decreasing)
- 📉 Moving averages (3-year MA)
- 📍 Min/max value detection

### 3. **Statistical Testing**
- 🔬 Augmented Dickey-Fuller (ADF) test for stationarity
- 📐 Chi-square test for independence
- 📊 Two-way ANOVA for state/year comparison
- 📈 Autocorrelation Function (ACF) analysis
- 📉 Partial Autocorrelation Function (PACF) analysis

### 4. **ARIMA Modeling**
- 🤖 Automated ARIMA model fitting
- 🔮 3-year forecasting (2013-2015)
- 📊 95% confidence intervals
- 📉 Residual diagnostics
- 📈 Model performance metrics (AIC, BIC)

### 5. **Crime Index Calculation**
- 🎯 Base year comparison (2001)
- 🚦 Safety classification:
  - **SAFE:** Index < 100
  - **MODERATE:** 100 ≤ Index < 120
  - **NOT SAFE:** Index ≥ 120

### 6. **Visualizations**
- 📊 Time series plots with forecasts
- 📈 Moving average trend lines
- 📉 ACF/PACF correlation plots
- 🎨 Comparative state analysis
- 📊 Crime index bar charts
- 🗺️ Multi-panel analysis dashboards

---

## 📂 Project Structure

```
crime-arima-analysis/
│
├── crime_arima_analysis.py          # Main analysis script
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── LICENSE                           # License file
│
├── data/
│   └── Crime Data Insights.pdf       # Source data document
│
├── outputs/
│   ├── Chhattisgarh_MURDER_ARIMA.png
│   ├── Gujarat_KIDNAPPING_ARIMA.png
│   ├── Maharashtra_RIOTS_ARIMA.png
│   ├── Comparative_Analysis_All_States.png
│   ├── Crime_Index_Analysis.png
│   └── Crime_Analysis_Summary_Report.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # Jupyter notebook for exploration
│
└── docs/
    ├── methodology.md                # Detailed methodology
    └── findings.md                   # Key findings and insights
```

---

## 📊 Output Files

### Generated Visualizations
1. **Individual State Analysis Plots** (`{State}_{Crime}_ARIMA.png`)
   - Time series with forecast
   - ACF plot
   - PACF plot
   - Residual analysis

2. **Comparative Analysis** (`Comparative_Analysis_All_States.png`)
   - Multi-panel comparison across all states
   - All 6 crime types visualized

3. **Crime Index Chart** (`Crime_Index_Analysis.png`)
   - Bar chart showing safety status of each state
   - Color-coded by safety level

### Generated Reports
1. **Summary Report** (`Crime_Analysis_Summary_Report.csv`)
   - State-wise crime statistics
   - Trend analysis
   - Forecast predictions
   - Percentage change calculations

---

## 🔍 Methodology

### 1. Data Collection
- Extraction from official crime records (2001-2012)
- Data validation and consistency checks
- Time series structuring

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics calculation
- Trend identification
- Outlier detection
- Correlation analysis

### 3. Stationarity Testing
- **Augmented Dickey-Fuller (ADF) Test**
  - Null Hypothesis (H₀): Series is non-stationary
  - Alternative Hypothesis (H₁): Series is stationary
  - Significance level: α = 0.05

### 4. ARIMA Model Selection
- **Model Order:** ARIMA(p, d, q)
  - p: AutoRegressive order
  - d: Differencing order
  - q: Moving Average order
- **Parameter Optimization:** Grid search or auto-ARIMA
- **Model Validation:** AIC, BIC criteria

### 5. Forecasting
- 3-year ahead predictions (2013-2015)
- 95% confidence intervals
- Forecast accuracy assessment

### 6. Crime Index Calculation
```
Crime Index = (Crime_2012 / Crime_2001) × 100
Average Crime Index = Mean of all crime indices
```

---

## 📈 Key Findings

### State-wise Safety Assessment

| State | Crime Index | Safety Status |
|-------|-------------|---------------|
| Rajasthan | 76.3 - 96.9 | **SAFE** |
| Chhattisgarh | 124.3 - 247.8 | **NOT SAFE** |
| Gujarat | 144.3 - 175.2 | **NOT SAFE** |
| Haryana | 83.7 - 169.7 | **SAFE to NOT SAFE** |
| Madhya Pradesh | 108.3 - 114.1 | **MODERATE** |
| Maharashtra | 94.4 - 158.3 | **SAFE to NOT SAFE** |

### Crime Trends (2001-2012)

#### Increasing Trends
- ✅ **Kidnapping:** Sharp increase across all states (especially Haryana, Gujarat)
- ✅ **Rape:** Gradual increase in most states
- ✅ **Murder:** Slight increase in Haryana

#### Decreasing Trends
- ✅ **Riots:** Significant decrease in Rajasthan, Gujarat
- ✅ **Robbery:** Fluctuating but generally stable

#### Stable Trends
- ✅ **Counterfeiting:** Relatively constant across states

### ARIMA Forecast Insights (2013-2015)
- **Murder:** Expected to stabilize or decrease slightly
- **Kidnapping:** Predicted to continue increasing
- **Riots:** Forecasted to maintain decreasing trend
- **Rape:** Expected gradual increase requiring intervention

---

## 💡 Actionable Insights

### For Law Enforcement
1. **Resource Allocation**
   - Increase patrols in high-risk areas (kidnapping hotspots)
   - Deploy specialized units for rising crime categories

2. **Proactive Policing**
   - Use forecasts to anticipate crime surges
   - Implement preventive measures during high-risk periods

3. **Crime Prevention**
   - Focus on kidnapping prevention in Gujarat and Haryana
   - Address increasing rape cases through awareness campaigns

### For Policy Makers
1. **Strategic Planning**
   - Allocate budgets based on crime predictions
   - Develop state-specific crime prevention policies

2. **Intervention Programs**
   - Launch targeted programs in "NOT SAFE" states
   - Strengthen judicial processes for deterrence

3. **Infrastructure Development**
   - Improve surveillance in crime-prone areas
   - Enhance emergency response systems

---

## 🔬 Statistical Tests Performed

### 1. Chi-Square Test
- **Purpose:** Test independence between states and crime types
- **Result:** Significant relationship found (χ² = 6841.61, p < 0.05)

### 2. Two-Way ANOVA
- **Factors:** Years (2001-2012) and States (6)
- **Results:**
  - Crime rates are **similar across years** (F₁ = 1.16, p > 0.05)
  - Crime rates are **different across states** (F₂ = 95.44, p < 0.05)

### 3. Stationarity Tests (ADF)
- Most time series required differencing (d=1)
- After differencing, series became stationary

---

## 📚 References

### Academic Sources
1. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*
3. Crime Records Bureau, India - Official Statistics

### Data Sources
- [Kaggle - Crime in India Dataset](https://www.kaggle.com/datasets/rajanand/crime-in-india)
- National Crime Records Bureau (NCRB), India

### Libraries Documentation
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Add more states to the analysis
- [ ] Implement advanced forecasting models (SARIMA, Prophet)
- [ ] Create interactive dashboards (Plotly, Dash)
- [ ] Add machine learning classification models
- [ ] Develop API for real-time predictions

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Avisikta Das**
- 🎓 B.Sc. Statistics (Honours), Semester 6
- 🏫 Asutosh College, University of Calcutta
- 📧 Email: [your.email@example.com]
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐙 GitHub: [@yourusername]

### Supervisor
**Prof. Dr. Shirsendu Mukherjee**
- 🏫 Assistant Professor, Department of Statistics
- 🎓 Asutosh College, University of Calcutta

---

## 🙏 Acknowledgments

Special thanks to:
- **Prof. Dr. Shirsendu Mukherjee** for guidance and supervision
- **Dr. Manas Kabi**, Principal, Asutosh College
- **Prof. Dr. Dhiman Dutta**, Head of Department, Statistics
- **Department of Statistics**, Asutosh College for support and resources
- **National Crime Records Bureau** for making crime data publicly available

---

## 📞 Contact

For questions, suggestions, or collaborations:

- 📧 Email: statistics.asutosh@college.edu.in
- 🏛️ Address: Department of Statistics, Asutosh College
  - 92, Shyama Prasad Mukherjee Road
  - Jatin Das Park, Patuapara, Bhowanipore
  - Kolkata, West Bengal 700026, India

---

## 📊 Project Status

- ✅ Data Collection: **Complete**
- ✅ Data Preprocessing: **Complete**
- ✅ Exploratory Analysis: **Complete**
- ✅ ARIMA Modeling: **Complete**
- ✅ Forecasting: **Complete**
- ✅ Visualization: **Complete**
- ✅ Documentation: **Complete**
- 🔄 Future Enhancements: **In Progress**

---

## 🎯 Future Enhancements

### Short-term
- [ ] Add more recent data (2013-2024)
- [ ] Implement SARIMA for seasonal patterns
- [ ] Create interactive Jupyter notebooks
- [ ] Add unit tests for functions

### Long-term
- [ ] Develop web-based dashboard (Streamlit/Dash)
- [ ] Integrate machine learning classification models
- [ ] Add geospatial visualization with maps
- [ ] Create API for real-time crime predictions
- [ ] Expand analysis to all Indian states

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{das2023crime,
  author = {Das, Avisikta},
  title = {Crime Data Analysis Using ARIMA: Western & Middle India},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/crime-arima-analysis}},
  note = {Supervised by Prof. Dr. Shirsendu Mukherjee, Asutosh College}
}
```

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ by Avisikta Das | Department of Statistics, Asutosh College

</div>
