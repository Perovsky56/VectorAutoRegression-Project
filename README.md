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
