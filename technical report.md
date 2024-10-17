# Reservoir Inflow Prediction Using Data-Driven Methods
## Technical Report

### 1. Data Collection and Processing

#### 1.1 Data Sources
The following data was collected from specified online sources:
- Daily Temperature Data (°C) - [temperature site](https://www.visualcrossing.com/weather-history/nalgonda)
- Daily Humidity Data (%) - [temperature site](https://www.visualcrossing.com/weather-history/nalgonda)
- Daily Solar Energy Data (MJ/m²) - [temperature site](https://www.visualcrossing.com/weather-history/nalgonda)
- Daily Rainfall Data (mm) - [wris site](https://indiawris.gov.in/wris/#/)
- Daily Soil-Moisture Data (%) - [wris site](https://indiawris.gov.in/wris/#/)
- Daily Evapotranspiration Data (mm) - [wris site](https://indiawris.gov.in/wris/#/)
- Daily Reservoir Inflow Data (BCM, later converted to m³/s) - [wris site](https://indiawris.gov.in/wris/#/)

The study focused on the Krishna Middle Basin in Telangana, with data collection attempted from July 2018 onwards.

#### 1.2 Data Preprocessing
Two main preprocessing steps were implemented:

**Interpolation:**
- Mathematical technique used to estimate unknown values
- Applied to handle missing/NaN values in all collected datasets

**Outlier Detection:**
- Comprehensive outlier analysis performed
- Boxplots generated for each dataset to visualize and identify anomalies

#### 1.3 Data Split
The consolidated dataset was divided into three sets:
- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

### 2. Model Performance Results

#### 2.1 Time Series Models

**Facebook Prophet**
- Mean Absolute Error (MAE): 2,985.04

**LSTM (Long Short-Term Memory)**
- Mean Absolute Error (MAE): 2,670.81

#### 2.2 Regression Models

**Random Forest Regression**
- Root Mean Squared Error (RMSE): 9,489.32
- Feature Importance:
  - Evapotranspiration: 0.251718
  - Temperature: 0.173066
  - Soil Moisture: 0.146909
  - Solar Energy: 0.146271
  - Humidity: 0.145318
  - Rainfall: 0.136718

**Ridge Regression**
- Root Mean Squared Error (RMSE): 8,163.50

**Multilayer Feed Forward Neural Network**
- Mean Absolute Error (MAE): 2,521.50
- Best Configuration:
  - Hidden Layer 1: 653 neurons
  - Hidden Layer 2: 661 neurons
  - Activation Function: Tanh
  - Solver: Adam
  - Maximum Iterations: 938

**Gradient Boosting Regression**
- Mean Absolute Error (MAE): 3,711.60

### 3. Final Model Selection and Conclusion

Based on the performance metrics, the Multilayer Feed Forward Neural Network emerged as the best-performing model with the lowest Mean Absolute Error of 2,521.50. This performance can be attributed to its optimal architecture determined through hyperparameter tuning using Optuna.

The model comparison reveals a clear hierarchy in performance:
1. Multilayer Feed Forward Neural Network (MAE: 2,521.50)
2. LSTM (MAE: 2,670.81)
3. Facebook Prophet (MAE: 2,985.04)
4. Gradient Boosting Regression (MAE: 3,711.60)

The Random Forest and Ridge Regression models, while providing valuable insights through feature importance analysis, showed higher error rates compared to the neural network approaches.

For practical implementation, the Multilayer Feed Forward Neural Network is recommended for reservoir inflow prediction in the Krishna Middle Basin, given its superior predictive accuracy.
