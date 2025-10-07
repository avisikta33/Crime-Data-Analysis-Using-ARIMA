"""
Crime Data Analysis Using ARIMA
Analysis of Western & Middle India Crime Data (2001-2012)
Mentor: Prof. Shirshendu Mukherjee, Asutosh College, University of Calcutta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== DATA EXTRACTION ====================
print("="*70)
print("CRIME DATA ANALYSIS USING ARIMA")
print("Western & Middle India (2001-2012)")
print("="*70)

# Crime data from PDF
crime_data = {
    'Chhattisgarh': {
        'MURDER': [880, 844, 776, 927, 1013, 1098, 1097, 1169, 1083, 1065, 1110, 998],
        'RAPE': [959, 992, 898, 969, 990, 995, 982, 978, 976, 1012, 1053, 1034],
        'KIDNAPPING': [207, 178, 177, 214, 246, 239, 244, 273, 286, 359, 472, 450],
        'ROBBERY': [338, 326, 331, 363, 401, 439, 427, 500, 554, 552, 470, 412],
        'RIOTS': [871, 817, 703, 790, 893, 905, 881, 1144, 957, 1092, 934, 910],
        'COUNTERFEITING': [8, 18, 24, 42, 73, 33, 64, 71, 68, 59, 65, 53]
    },
    'Gujarat': {
        'MURDER': [1226, 1532, 1114, 1113, 1033, 1165, 1166, 1106, 1020, 1048, 1126, 1126],
        'RAPE': [286, 267, 236, 339, 324, 354, 316, 374, 433, 408, 439, 473],
        'KIDNAPPING': [998, 1015, 1044, 1133, 1164, 1128, 1312, 1323, 1348, 1447, 1614, 1720],
        'ROBBERY': [991, 1098, 1267, 1136, 966, 970, 1095, 1322, 1420, 1384, 1368, 1366],
        'RIOTS': [1930, 3665, 1824, 1599, 1628, 1534, 1668, 1809, 1539, 1623, 1615, 1758],
        'COUNTERFEITING': [55, 46, 275, 122, 447, 196, 286, 190, 238, 255, 256, 216]
    },
    'Haryana': {
        'MURDER': [781, 760, 702, 733, 784, 873, 911, 921, 948, 1005, 1062, 991],
        'RAPE': [398, 361, 353, 386, 461, 608, 488, 631, 603, 720, 733, 668],
        'KIDNAPPING': [449, 427, 388, 423, 492, 635, 801, 854, 916, 963, 959, 1349],
        'ROBBERY': [397, 330, 281, 244, 390, 410, 502, 555, 679, 734, 638, 711],
        'RIOTS': [750, 700, 639, 699, 899, 1142, 1173, 1184, 1166, 1414, 1466, 1424],
        'COUNTERFEITING': [37, 32, 30, 32, 31, 35, 33, 40, 35, 29, 18, 20]
    },
    'Madhya Pradesh': {
        'MURDER': [2425, 2395, 2227, 2379, 2405, 2309, 2244, 2322, 2386, 2423, 2511, 2373],
        'RAPE': [2851, 2891, 2738, 2875, 2921, 2900, 3010, 2937, 2998, 3135, 3406, 3425],
        'KIDNAPPING': [956, 964, 887, 832, 847, 808, 922, 929, 1036, 1187, 1288, 1302],
        'ROBBERY': [1764, 1829, 1578, 1526, 1557, 1770, 1975, 2234, 2270, 1919, 1952, 1940],
        'RIOTS': [3395, 3178, 2729, 2479, 2573, 2308, 2648, 2768, 2409, 2606, 2080, 1951],
        'COUNTERFEITING': [19, 54, 38, 36, 51, 38, 38, 47, 27, 33, 12, 31]
    },
    'Maharashtra': {
        'MURDER': [2839, 2768, 2772, 2696, 2621, 2656, 2693, 2795, 2653, 2744, 2818, 2712],
        'RAPE': [1302, 1352, 1268, 1388, 1545, 1500, 1451, 1558, 1483, 1599, 1701, 1839],
        'KIDNAPPING': [985, 953, 926, 1045, 1194, 1261, 1312, 1379, 1286, 1508, 1669, 1583],
        'ROBBERY': [2239, 2136, 2096, 2229, 2411, 2574, 2770, 3031, 3314, 3721, 4249, 6949],
        'RIOTS': [6719, 6360, 5714, 5980, 6779, 7453, 7993, 9388, 8030, 8412, 8556, 8860],
        'COUNTERFEITING': [265, 176, 261, 218, 385, 317, 270, 405, 478, 389, 351, 291]
    },
    'Rajasthan': {
        'MURDER': [1259, 1276, 1262, 1279, 1221, 1209, 1303, 1297, 1395, 1421, 1461, 1461],
        'RAPE': [1049, 1051, 1050, 1038, 993, 1085, 1238, 1355, 1519, 1571, 1800, 2049],
        'KIDNAPPING': [2718, 2587, 2238, 2384, 1993, 1970, 2177, 2358, 2870, 2985, 3204, 3243],
        'ROBBERY': [889, 734, 655, 673, 590, 631, 778, 829, 886, 872, 727, 807],
        'RIOTS': [11214, 7178, 4052, 3101, 2290, 1767, 1626, 1390, 1145, 986, 751, 573],
        'COUNTERFEITING': [89, 63, 58, 65, 88, 56, 56, 51, 59, 36, 56, 49]
    }
}

years = list(range(2001, 2013))

# ==================== HELPER FUNCTIONS ====================

def check_stationarity(timeseries, title):
    """Perform Augmented Dickey-Fuller test"""
    print(f"\n{title}")
    print("-" * 50)
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("✓ Series is STATIONARY (Reject H0)")
    else:
        print("✗ Series is NON-STATIONARY (Accept H0)")
    return result[1] <= 0.05

def calculate_crime_index(state_data, base_year_idx=0, current_year_idx=11):
    """Calculate crime index based on base year"""
    indices = []
    for crime_type, values in state_data.items():
        if values[base_year_idx] > 0:
            index = (values[current_year_idx] / values[base_year_idx]) * 100
            indices.append(index)
    avg_index = np.mean(indices)
    
    if avg_index < 100:
        status = "SAFE"
    elif avg_index < 120:
        status = "MODERATE"
    else:
        status = "NOT SAFE"
    
    return avg_index, status

def moving_average(data, window=3):
    """Calculate moving average"""
    return pd.Series(data).rolling(window=window, center=True).mean()

# ==================== DESCRIPTIVE STATISTICS ====================

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

for state in crime_data.keys():
    print(f"\n{state.upper()}")
    print("-" * 50)
    
    state_df = pd.DataFrame(crime_data[state], index=years)
    print(state_df.describe().round(2))
    
    # Crime Index
    crime_index, safety_status = calculate_crime_index(crime_data[state])
    print(f"\nCrime Index (2012 vs 2001): {crime_index:.2f}")
    print(f"Safety Status: {safety_status}")

# ==================== ARIMA MODELING ====================

def fit_arima_model(data, order=(1,1,1), forecast_periods=3):
    """Fit ARIMA model and generate forecasts"""
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        # Get confidence intervals
        forecast_df = fitted_model.get_forecast(steps=forecast_periods)
        conf_int = forecast_df.conf_int()
        
        return fitted_model, forecast, conf_int
    except Exception as e:
        print(f"Error fitting ARIMA: {e}")
        return None, None, None

# ==================== ANALYSIS FOR EACH STATE ====================

def analyze_state(state_name, crime_type):
    """Complete ARIMA analysis for a specific state and crime type"""
    
    print("\n" + "="*70)
    print(f"ARIMA ANALYSIS: {crime_type} in {state_name}")
    print("="*70)
    
    # Get data
    data = crime_data[state_name][crime_type]
    ts_data = pd.Series(data, index=years)
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Std Dev: {np.std(data):.2f}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Trend: {'Increasing' if data[-1] > data[0] else 'Decreasing'}")
    
    # Check stationarity
    is_stationary = check_stationarity(ts_data, "Stationarity Test")
    
    # Fit ARIMA model
    print("\nFitting ARIMA Model...")
    model, forecast, conf_int = fit_arima_model(data, order=(1,1,1), forecast_periods=3)
    
    if model is not None:
        print(f"\nModel Summary:")
        print(f"AIC: {model.aic:.2f}")
        print(f"BIC: {model.bic:.2f}")
        
        print(f"\nForecast for 2013-2015:")
        for i, (year, pred) in enumerate(zip(range(2013, 2016), forecast)):
            lower = conf_int.iloc[i, 0]
            upper = conf_int.iloc[i, 1]
            print(f"{year}: {pred:.0f} (95% CI: [{lower:.0f}, {upper:.0f}])")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{crime_type} Analysis - {state_name}', fontsize=16, fontweight='bold')
    
    # 1. Time Series Plot
    axes[0, 0].plot(years, data, marker='o', linewidth=2, label='Actual')
    axes[0, 0].plot(years, moving_average(data, 3), 
                    linestyle='--', linewidth=2, label='3-Year MA')
    if forecast is not None:
        forecast_years = list(range(2013, 2016))
        axes[0, 0].plot(forecast_years, forecast, 
                       marker='s', linestyle='--', linewidth=2, 
                       color='red', label='Forecast')
        axes[0, 0].fill_between(forecast_years, 
                               conf_int.iloc[:, 0], 
                               conf_int.iloc[:, 1], 
                               alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Cases')
    axes[0, 0].set_title('Time Series with Forecast')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ACF Plot
    plot_acf(ts_data, lags=10, ax=axes[0, 1])
    axes[0, 1].set_title('Autocorrelation Function (ACF)')
    
    # 3. PACF Plot
    plot_pacf(ts_data, lags=10, ax=axes[1, 0])
    axes[1, 0].set_title('Partial Autocorrelation Function (PACF)')
    
    # 4. Residuals
    if model is not None:
        residuals = model.resid
        axes[1, 1].plot(residuals, marker='o')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Observation')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{state_name}_{crime_type}_ARIMA.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{state_name}_{crime_type}_ARIMA.png'")
    
    return model, forecast

# ==================== COMPARATIVE ANALYSIS ====================

def comparative_analysis():
    """Compare all states"""
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS - ALL STATES")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Crime Trends Across States (2001-2012)', fontsize=16, fontweight='bold')
    
    crime_types = ['MURDER', 'RAPE', 'KIDNAPPING', 'ROBBERY', 'RIOTS', 'COUNTERFEITING']
    
    for idx, crime_type in enumerate(crime_types):
        ax = axes[idx // 3, idx % 3]
        
        for state in crime_data.keys():
            data = crime_data[state][crime_type]
            ax.plot(years, data, marker='o', label=state, linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Cases')
        ax.set_title(crime_type)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Comparative_Analysis_All_States.png', dpi=300, bbox_inches='tight')
    print("\nComparative plot saved as 'Comparative_Analysis_All_States.png'")

# ==================== CRIME INDEX VISUALIZATION ====================

def visualize_crime_indices():
    """Visualize crime indices for all states"""
    
    print("\n" + "="*70)
    print("CRIME INDEX ANALYSIS")
    print("="*70)
    
    states = list(crime_data.keys())
    indices = []
    statuses = []
    
    for state in states:
        index, status = calculate_crime_index(crime_data[state])
        indices.append(index)
        statuses.append(status)
        print(f"{state}: {index:.2f} - {status}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if s == 'SAFE' else 'orange' if s == 'MODERATE' else 'red' 
              for s in statuses]
    bars = ax.bar(states, indices, color=colors, alpha=0.7, edgecolor='black')
    
    # Add reference lines
    ax.axhline(y=100, color='green', linestyle='--', label='Safe Threshold')
    ax.axhline(y=120, color='orange', linestyle='--', label='Moderate Threshold')
    
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Crime Index', fontsize=12)
    ax.set_title('Crime Index by State (Base Year: 2001)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, idx in zip(bars, indices):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{idx:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Crime_Index_Analysis.png', dpi=300, bbox_inches='tight')
    print("\nCrime index plot saved as 'Crime_Index_Analysis.png'")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # Example: Detailed analysis for specific cases
    print("\n" + "="*70)
    print("DETAILED ARIMA ANALYSIS")
    print("="*70)
    
    # Analyze Murder in Chhattisgarh
    analyze_state('Chhattisgarh', 'MURDER')
    
    # Analyze Kidnapping in Gujarat
    analyze_state('Gujarat', 'KIDNAPPING')
    
    # Analyze Riots in Maharashtra
    analyze_state('Maharashtra', 'RIOTS')
    
    # Comparative Analysis
    comparative_analysis()
    
    # Crime Index Visualization
    visualize_crime_indices()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Outputs Generated:")
    print("1. Individual ARIMA analysis plots for selected states/crimes")
    print("2. Comparative analysis across all states")
    print("3. Crime index visualization")
    print("4. Statistical summaries and forecasts")
    print("\nAll visualizations saved as PNG files in the current directory.")
    
    # Show plots
    plt.show()

# ==================== ADDITIONAL ANALYSIS FUNCTIONS ====================

def generate_summary_report():
    """Generate a comprehensive summary report"""
    
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    report = []
    
    for state in crime_data.keys():
        for crime_type in crime_data[state].keys():
            data = crime_data[state][crime_type]
            
            # Fit ARIMA
            model, forecast, conf_int = fit_arima_model(data)
            
            if forecast is not None:
                trend = "Increasing" if data[-1] > data[0] else "Decreasing"
                avg_forecast = np.mean(forecast)
                
                report.append({
                    'State': state,
                    'Crime': crime_type,
                    'Mean (2001-2012)': np.mean(data),
                    'Trend': trend,
                    'Forecast 2013-2015 (Avg)': avg_forecast,
                    'Change %': ((data[-1] - data[0]) / data[0] * 100)
                })
    
    report_df = pd.DataFrame(report)
    print("\n", report_df.to_string(index=False))
    
    # Save to CSV
    report_df.to_csv('Crime_Analysis_Summary_Report.csv', index=False)
    print("\n✓ Summary report saved as 'Crime_Analysis_Summary_Report.csv'")
    
    return report_df

# Generate summary report
generate_summary_report()

print("\n" + "="*70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("Mentor: Prof. Shirshendu Mukherjee")
print("Department of Statistics, Asutosh College")
print("University of Calcutta")
print("="*70)