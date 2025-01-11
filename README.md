# Project overview

This project focuses on the use of the VectorAutoRegression (VAR) model to forecast the average energy parameters of buildings based on hourly data. This project was born out of a desire to improve knowledge of the VAR statistical model - to explore its usefulness and effectiveness in capturing relationships between multiple variables that change over time.

These data are sourced from the government website https://www.gov.pl/web/archiwum-inwestycje-rozwoj/dane-do-obliczen-energetycznych-budynkow


# Tools used

- **Python:**
  - **pandas:**  for convenient manipulation and analysis of tabular data, such as DataFrames
  - **numpy:** for handling arrays and performing efficient mathematical and numerical operations
  - **matplotlib:** for creating aesthetic and transparent plots and data visualizations
  - **statsmodel:** for statistical and econometric modeling - essential for projects involving VAR
  - **sklearn.metrics:** providing model evaluation metrics, such as accuracy, precision, and regression measures
- **Jupyter Notebooks:** an environment for writing and running Python code, combining code, visualizations, and rich text in a single document
- **Git & GitHub:** essential for version control and sharing code, ensuring collaboration and project tracking
- **Excel:** a spreadsheet application for organizing and analyzing data
- **Overleaf:** an online LaTeX editor that allows writing, editing, and compiling of LaTeX documents in real-time, with no need for local installations


# Dataset overview

## Meteorological stations
The data source includes a long list of meteorological stations across the Polish country, from which it was possible to select the best ones to study the topic. 

The selection took into account the distance between the various stations so that they could be far apart, as well as the quality of the data that specific sets have - including whether they contain all the desired features, whether they do not have many NaN values, and whether the data are collected for a large number of years.

Selected cities and meteorological stations were: Kołobrzeg, Opole and Warszawa(Okęcie).


## Features
Databases for each station include the following features:
- Hour of the year (N): Sequential hour number within the year,
- Month (M): Month of the year,
- Day (D): Day of the month,
- Hour UTC (H): Hour in Coordinated Universal Time (UTC),
- Dry bulb temperature (DBT): Temperature measured by a dry thermometer in degrees Celsius (°C),
- Relative humidity (RH): Percentage of moisture inthe air relative to the maximum amount the air can hold at that temperature,
- Humidity ratio (HR): Mass of water vapor per unitmass of dry air, measured in grams per kilogram (g/kg),
- Wind speed (WS): Speed of the wind in meters per second (m/s),
- Wind direction (WD): Direction from which the wind is blowing, categorized into 36 sectors (0 - calm, N - 36, E - 9, S - 18, W - 27, 99 - variable),
- Total solar radiation on a horizontal surface (ITH): Total solar energy received per unit area on a horizontal surface, measured in watts per square meter (W/m²),
- Direct solar radiation on a horizontal surface(IDH): Solar radiation received directly from the sun on a horizontal surface, measured in (W/m²),
- Diffuse solar radiation on a horizontal surface(ISH): Solar radiation received from the sky (excluding direct sunlight) on a horizontal surface, measured in (W/m²),
- Sky radiation temperature (TSKY): Temperature of the sky as perceived by a radiometer, measured in degrees Celsius (°C),
- Total solar radiation intensity per horizontal surface (direction N inclination 0º) (N_0) in (W/m²),
- Total solar irradiance on surfaces with N, NE, E, SE, S, SW, W, NW orientation and inclination to horizontal 30º, 45º, 60º, 90º (N_30, NE_30, ...) in (W/m²) which includes 32 features.


# Dataset processing and cleaning

The collected data had to be first imported and then cleaned to be ready for use in the VAR model

```python
# all of the features before cleaning
columns = [
    'N', 'M', 'D', 'H', 'DBT', 'RH', 'HR', 'WS', 'WD',
    'ITH', 'IDH', 'ISH', 'TSKY',
    'N_0', 'N_30', 'NE_30', 'E_30', 'SE_30', 'S_30', 'SW_30', 'W_30', 'NW_30',
    'N_45', 'NE_45', 'E_45', 'SE_45', 'S_45', 'SW_45', 'W_45', 'NW_45',
    'N_60', 'NE_60', 'E_60', 'SE_60', 'S_60', 'SW_60', 'W_60', 'NW_60',
    'N_90', 'NE_90', 'E_90', 'SE_90', 'S_90', 'SW_90', 'W_90', 'NW_90'
]

# converting all values to numeric, and the error ones to NaN
df = df.drop(index=0)
df = df.apply(pd.to_numeric, errors='coerce')

# function calculating the date from features for each row passed
def calculate_date(row):
    total_hours = row['N'] - 1
    remaining_hours = total_hours % 8760

    days_passed = remaining_hours // 24
    hour = row['H']

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    month = 0
    while days_passed >= month_days[month]:
        days_passed -= month_days[month]
        month += 1

    day = days_passed + 1
    return pd.Timestamp(year=1992, month=int(month) + 1, day=int(day), hour=int(hour))

# applying the function for every row and changing it to better format after all
df['date'] = df.apply(calculate_date, axis=1)
df['date'] = df['date'].dt.strftime('%m-%d %H:00')
df.set_index('date', inplace=True)
```


# Methodology

A Vector Autoregression (VAR) model was employed using the ‘statsmodels‘ library. The model selection was based on the Akaike Information Criterion (AIC), which helps in selecting the model with the best fit by balancing the complexity and goodness of fit. 
The maximum number of lags considered was 50, ensuring that the model captures the temporal dependencies adequately

During testing, it was deduced that features containing information on total radiation intensity should be discarded, among other reasons:
- frequent empty values in the database,
- the too little influence on the prediction of other features.
This also effectively affected the cost of the model and the time it took to make predictions.

Any columns containing time information were also not included in the features selected to make predictions.


## Stationarity

**Stationarity** refers to the property of a time series that remains constant over time in terms of its statistical properties. This is a key condition in time series analysis.

A stochastic process is stationary if its key statistics (such as mean, variance and autocorrelation) do not change over time. So, **no seasonality and no trend are noticeable**.

When the features are non-stationary, this can make it much more difficult to find relationships between them and how they affect each other - instead, they are determined by the passage of time and time trends, which will not allow effective prediction of the future.

One method for testing the stationarity of time series is the **Augmented Dickey-Fuller (ADF) test**, confirming its suitability for the VAR model.

```python
# Testing each timeseries for stationarity and visualizing each on a chart
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

for column in selected_columns:
    print(f'Testing {column} for stationarity:')
    test_stationarity(df_selected[column])
    df[column].plot(kind='line', title=column, linewidth=0.5, figsize=(15,6))
    plt.xticks(rotation=90)
    plt.show()
```


### Two examples of AFD test performed on two features
- Column featuring DBT (Dry bulb temperature)
![ADF_DBT](https://github.com/user-attachments/assets/8612cee1-7851-4e15-8a97-e3062c6f7a79)

- Column featuring WS (Wind speed)
![ADF_WS](https://github.com/user-attachments/assets/f82b20dc-befc-48d7-82d5-733a8e0cad11)

### Insights
The ADF test results for the DBT and WS timeseries were specifically chosen to show how they differ in terms of stationarity.

The results of the ADF test should be compared with the critical values, which are the thresholds by which it is determined with how much certainty a time series can be determined stationary.

For the DBT feature, the test value is -2.39, which, compared to the critical values, is above the 10% threshold (-2.39 > -2.57) - this means very low confidence that this time series is stationary. With this, it is inferred that **DBT timeseries is non-stationary**.

For the WS feature, the test value is -9.15, which, compared to the critical values, is deep below the 1% threshold - this means more than 99 percent confidence that the **WS timeseries is stationary**.

The graphs make it possible, using the eye test, to see that indeed dry bulb temperature shows a trend and seasonality, while wind speed is completely independent of the passage of time.


## Calculations

The calculations were performed using the VAR modelfrom ‘statsmodels‘. The analysis considered 5000 hours of the year, and the model was used to forecast the next 24 hours. This approach allows for understanding the shortterm dynamics and dependencies between the variables over a one-day horizon

In the VAR (Vector Autoregressive) model, each feature in the system is described as a function of its past values (lagged observations) and the past values of other features. This means that the variables are not treated as independent, but as interrelated over time, and their changes affect each other.

## Evaluation Metrics

The prediction accuracy was evaluated by comparing the actual values of the next 24 hours with the forecasted values using the following metrics:
- **Mean Squared Error (MSE):** This metric measures the average of the squares of the errors, indicating the average squared difference between the estimated values and the actual value,
- **Mean Absolute Error (MAE):** This metric measures the average magnitude of the errors in a set of predictions, without considering their direction,
- **R-squared (R²):** This metric represents the proportion of the variance for a dependent variable that’s explained by an independent variable or variables in a regression model

# Results

The results of the VAR model predictions for one selected city (Kołobrzeg) are presented in this section. The predictions include dry bulb temperature (DBT), relative humidity (RH), and wind speed(WS) for the next 24 hours, based on the previous 100 hours of data

In a nutshell, the VAR statistical model involves making predictions that predict subsequent values of selected features, based on other variables. 
It is important to find such features for the VAR model, that are related to others and have a strong correlation with them.

***Correlation matrix of residuals***
             DBT        RH        HR        WS        WD       ITH       IDH       ISH      TSKY
DBT     1.000 -0.595  0.165  0.018 -0.024  0.104  0.031  0.092  0.370
RH     -0.595  1.000  0.593 -0.011 -0.010 -0.095 -0.026 -0.086 -0.143
HR      0.165  0.593  1.000  0.002 -0.032 -0.020 -0.015 -0.008  0.125
WS      0.018 -0.011  0.002  1.000  0.095 -0.004  0.006 -0.011  0.046
WD     -0.024 -0.010 -0.032  0.095  1.000  0.008  0.027 -0.021  0.010
ITH     0.104 -0.095 -0.020 -0.004  0.008  1.000  0.631  0.511 -0.415
IDH     0.031 -0.026 -0.015  0.006  0.027  0.631  1.000 -0.345 -0.263
ISH     0.092 -0.086 -0.008 -0.011 -0.021  0.511 -0.345  1.000 -0.210
TSKY    0.370 -0.143  0.125  0.046  0.010 -0.415 -0.264 -0.210  1.000

**Evaluuation metrics matrix**
  Column        MSE      MAE      R2
0    DBT       4.79     1.55    0.69
1     RH     127.97    10.24    0.67
2     HR       1.48     1.07   -1.33
3     WS       2.11     1.29    0.16
4     WD     130.47    10.63   -0.06
5    ITH    8602.91    67.68    0.92
6    IDH   32323.88   111.52    0.55
7    ISH    9158.39    63.84   -0.04
8   TSKY      26.63     4.34   -4.09

### Plots of forecasts for DBT, RH, WS
![kolobrzeg_DBT](https://github.com/user-attachments/assets/052795c7-6b0c-44bf-9b3d-47d19971663d)

![kolobrzeg_RH](https://github.com/user-attachments/assets/8b74ae3d-28c6-42af-b1e0-9d4257863434)

![kolobrzeg_WS](https://github.com/user-attachments/assets/6fe0f4e6-013f-4417-b624-e6a65a95a988)


# Discussion of results

The results indicate that the VAR model performs reasonably well in predicting dry bulb temperature (DBT) and relative humidity (RH) across all three cities. However, the predictions for wind speed (WS) are less accurate, which can be attributed to the smaller changes and fewer data points available for this variable

In Kołobrzeg, the model achieved an R² of 0.69 for DBT and 0.67 for RH, indicating a good fit. However, the R² for WS was only 0.16, suggesting that the model struggled to capture the variability in wind speed.

Overall, the VAR model demonstrated its capability to predict temperature and humidity with reasonable accuracy, but improvements are needed for wind speed predictions. The correlation matrices suggest that there are significant interdependencies between the variables, which the model partially captures.

# Conclusions

The VAR model’s ability to handle multiple interrelated time series variables simultaneously is one of its key strengths. In this study, the model was applied to a comprehensive dataset that included not only DBT, RH, and WS but also other meteorological parameters such as humidity ratio (HR), wind direction (WD), total solar radiation (ITH), direct solar radiation (IDH), diffuse solar radiation (ISH), and sky radiation temperature (TSKY). This extensive dataset allowed the VAR model to capture the complex dynamics and interactions between these variables, providing a robust framework for forecasting.

An important aspect of the VAR model used in this study was its ability to automatically select the optimal number of lags (hours) to include in the prediction model. Although the model was initially set to consider up to 50 lags, it utilized the Akaike Information Criterion (AIC) to determine the most appropriate number of hours to look back for making accurate predictions. This adaptive feature ensured that the model was not overfitted and could generalize well to new data.

The analysis showed that the VAR model requires a wide range of correlated features to improve prediction accuracy. The inclusion of multiple variables helps the model to better understand the underlying relationships and dependencies, leading to more accurate forecasts.

Overall, the VAR model demonstrated its capability to predict temperature and humidity with reasonable accuracy, making it a valuable tool for building energy calculations. The model’s performance highlights the importance of having a wide range of correlated features to improve prediction accuracy. The VAR model requires a comprehensive dataset with multiple interrelated variables to fully capture the dynamics of the system. 

Future work could focus on enhancing the model’s ability to predict wind speed by incorporating additional data or using more advanced modeling approaches. Additionally, addressing the issue of missing data in the dataset could further improve the model’s performance. Exploring other robust methods and integrating more features could also enhance the predictive power of the VAR model.
