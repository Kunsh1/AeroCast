# AeroCast 🌬️

## Air Quality Index (AQI) Forecasting System using LSTM

This project develops an advanced Air Quality Index (AQI) forecasting system designed to provide accurate hourly predictions for urban areas. Leveraging historical data, real-time API feeds, and a Long Short-Term Memory (LSTM) neural network, the system aims to offer crucial insights for public health, environmental management, and daily planning.

## 🌟 Features

* **Historical Data Integration:** Processes aggregated and cleaned historical AQI and meteorological data.

* **Real-time API Integration:** Fetches live current AQI from the WAQI API and future weather forecasts from the OpenWeatherMap API to ensure up-to-date predictions.

* **LSTM Neural Network:** Utilizes a powerful LSTM model, specifically suited for time-series data, to learn complex temporal patterns.

* **Iterative 72-Hour Forecast:** Generates hourly AQI predictions for the next 72 hours using a rolling window approach, where each prediction informs the next.

* **Automated Data Preprocessing:** Includes steps for cleaning, feature engineering (time-based features), and scaling of data.

* **Model Persistence:** Saves the trained LSTM model and data scalers to avoid retraining on every run.

* **Performance Evaluation:** Reports key metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on a test set.

* **Comprehensive Visualization:** Generates an intuitive graph displaying the 72-hour forecast, AQI severity categories, and forecast uncertainty.

* **CSV Export:** Saves the generated 72-hour forecast into a CSV file for easy access and further analysis.

## 🛠️ Technologies Used

* **Programming Language:** Python 3.x

* **Machine Learning Framework:** TensorFlow 2.x / Keras

* **Data Manipulation:** Pandas, NumPy

* **Data Preprocessing & Metrics:** Scikit-learn

* **API Interaction:** Requests, Geopy

* **Visualization:** Matplotlib, Seaborn

* **Model Persistence:** Joblib

* **Development Environment:** Google Colab (recommended for GPU acceleration)

## 📊 Data Sources

* **Historical AQI Data (`AQI Dataset.csv`):** This is the primary historical dataset for model training. It was created by collecting raw hourly AQI data **specifically for Delhi** from the **CPCB (Central Pollution Control Board) website** for multiple years, combining it, and then performing initial cleaning and preprocessing. It includes historical AQI and various meteorological parameters.

* **Real-time Current AQI & Pollutant Forecasts:** **WAQI (World Air Quality Index) API**.

* **Future Meteorological Forecasts:** **OpenWeatherMap API** (Temperature, Humidity, Wind Speed, Pressure).

## 🧠 Model Architecture

The core of the forecasting system is a **Long Short-Term Memory (LSTM) neural network**. LSTMs are a type of Recurrent Neural Network (RNN) capable of learning from and remembering patterns in sequential data over long periods.

The model processes sequences of past `LOOK_BACK` hours (e.g., 24 hours), with each hour's input comprising AQI, various pollutant concentrations, meteorological conditions, and time-based features. The model is trained to predict the AQI for the subsequent hour.

## 📈 Performance

On the held-out test set, the LSTM model demonstrated strong predictive capabilities:

* **Mean Absolute Error (MAE):** 4.09

* **Root Mean Squared Error (RMSE):** 7.97

These metrics indicate that the model's predictions are, on average, very close to the actual AQI values.

## 🚀 Setup and Installation

1.  **Clone the repository:**

    ```
    git clone https://github.com/Kunsh1/AeroCast.git
    cd AeroCast
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```
    pip install pandas numpy scikit-learn matplotlib seaborn joblib requests geopy tensorflow
    ```

    *(Note: If using Google Colab, most of these are pre-installed. You might only need `geopy`.)*

4.  **Obtain API Keys:**

    * **OpenWeatherMap API Key:** Register at <https://openweathermap.org/> to get your API key.

    * **WAQI API Token:** Register at <https://aqicn.org/api/> to get your API token.

5.  **Prepare `AQI Dataset.csv`:** Ensure your preprocessed historical AQI data is available as `AQI Dataset.csv` in the project root directory. This CSV file should contain the following columns:

    * `DateTime` (Timestamp)

    * `AQI`

    * `PM2.5_24hr_avg`

    * `PM10_24hr_avg`

    * `NO2_24hr_avg`

    * `O3_8hr_avg`

    * `CO_8hr_avg`

    * `SO2_24hr_avg`

    * `NH3_24hr_avg`

    * `Temperature`

    * `Humidity`

    * `Wind_Speed`

    * `Solar_Radiation`

    * `Pressure`
        *(Note: The script will automatically derive `hour`, `day_of_week`, `month`, and `day_of_year` from the `DateTime` column.)*

6.  **Configure API Keys in the script:** Open the main Python script (`your_main_script_name.py` or the provided code block) and replace the placeholder API keys:

    ```
    OPENWEATHERMAP_API_KEY = 'YOUR_OPENWEATHERMAP_API_KEY'
    WAQI_API_TOKEN = 'YOUR_WAQI_API_TOKEN'
    ```

    Also, set your target `CITY_NAME`.

## 🏃 Usage

1.  **Run the main script:**

    ```
    python your_main_script_name.py
    ```

    *(If using Google Colab, simply run all cells in your notebook.)*

    **Note on Model Training:** The script is configured to automatically load a pre-trained model (`best_aqi_lstm_model.keras`) and scalers (`x_scaler.joblib`, `y_scaler.joblib`) if they exist in the same directory. If these files are not found, or if there's an error loading them, the model will be automatically trained from scratch. You can download the pre-trained model and scalers from the repository to skip the training step.

2.  **GPU Acceleration (Google Colab):**

    * Go to `Runtime` -> `Change runtime type`.

    * Select `GPU` as the `Hardware accelerator` and click `Save`.

## 📊 Results

Upon successful execution, the script will:

* Print evaluation metrics (MAE, RMSE) on the test set.

* Display a plot comparing actual vs. predicted AQI on the test set.

* Print the 72-hour hourly AQI forecast in the console.

* **Save the 72-hour forecast to `aqi_72hr_forecast.csv` in the project directory.**

* Display a detailed 72-hour forecast graph with AQI categories, uncertainty band, and hourly annotations.
