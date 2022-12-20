import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import sys

def get_data_summary(data):
    ROUND = 2
    summary = data.describe().to_dict()
    return {k: round(v, ROUND) for k, v in summary.items()}

def get_start_end_points_dict(data, count):
    start_end_points = {}
    
    start_end_points['start_indices'] = range(1, count + 1)
    start_end_points['start_data'] = data[:count]
    start_end_points['end_indices'] = range(len(data) - count, len(data))
    start_end_points['end_data'] = data[-count:]

    return start_end_points

def get_data_plot_image(data):
    fig, ax = plt.subplots(1, figsize=(15, 4))
    lenght = len(data)
    ax.plot(np.arange(lenght), data, c=np.random.choice(['olive', 'hotpink', 'turquoise', 'firebrick', 'navy', 'goldenrod']))
    ax.set_xticks(data.index)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.set_title("TIme Series Plot")

    image_file = StringIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)

    return image_file.getvalue()

def get_auto_arima_model(data):
    stdout = sys.stdout
    temp_out = StringIO()
    sys.stdout = temp_out

    auto_arima_model = auto_arima(
        data, 
        start_p=0,
        start_q=0,
        max_p=10,
        max_q=10,
        max_d=10,
        trace=True,
        test='adf',
        error_action='ignore',
        suppress_warnings=True,
        seasonal=False,
        stepwise=False
    )

    sys.stdout = stdout
    temp_out.seek(0)
    search_output = temp_out.read()

    model = ARIMA(data, order=auto_arima_model.get_params()['order']).fit()
    return model, search_output

def get_prediction_plot_image(data, model):
    NUM_OF_FORECASTS = 5
        
    lenght = len(data)
    predictions = model.predict(1, lenght + NUM_OF_FORECASTS - 1)

    conf = model.get_forecast(NUM_OF_FORECASTS - 1).conf_int(alpha=0.05)
    conf = np.insert(conf, 0, np.array([predictions[lenght - 1], predictions[lenght - 1]]), 0)

    forecast_index = np.arange(lenght - 1, lenght + NUM_OF_FORECASTS - 1)
    lower_series = pd.Series(conf[:, 0], index=forecast_index)
    upper_series = pd.Series(conf[:, 1], index=forecast_index)

    fig, ax = plt.subplots(1, figsize=(15,4))

    ax.plot(data, label='Actual')
    ax.plot(predictions, label='Predictions')
    ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    ax.set_title('Predictions vs Actuals')
    ax.legend(loc="upper left")

    image_file = StringIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)
    return image_file.getvalue()