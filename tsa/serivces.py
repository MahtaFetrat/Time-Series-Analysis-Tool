import sys
from io import StringIO, BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from tsa.models import TSADataset


def get_start_end_points_dict(data, count):
    start_end_points = {}
    count = min(count, len(data) // 2 - 1)

    start_end_points['start_indices'] = range(1, count + 1)
    start_end_points['start_data'] = data[:count]
    start_end_points['end_indices'] = range(len(data) - count + 1, len(data) + 1)
    start_end_points['end_data'] = data[-count:]

    return start_end_points


def create_data_plot_image(data):
    fig, ax = plt.subplots(1, figsize=(15, 4))
    lenght = len(data)
    ax.plot(np.arange(lenght), data,
            c=np.random.choice(['olive', 'hotpink', 'turquoise', 'firebrick', 'navy', 'goldenrod']))
    ax.set_xticks(data.index)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.set_title("TIme Series Plot")

    image_file = BytesIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)

    return image_file


def get_data_plot_image(tsa: TSADataset):
    key_name = 'data_plot'
    if not tsa.exist_image(key_name):
        image_data = create_data_plot_image(pd.Series(tsa.data_points))
        tsa.put_image(image_data, key_name)
    return tsa.url(key_name)


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


def create_prediction_plot_image(data, model):
    NUM_OF_FORECASTS = 5

    lenght = len(data)
    predictions = model.predict(1, lenght + NUM_OF_FORECASTS)

    conf = model.get_forecast(NUM_OF_FORECASTS).conf_int(alpha=0.05)
    conf = np.insert(conf, 0, np.array([predictions[lenght - 1], predictions[lenght - 1]]), 0)

    forecast_index = np.arange(lenght - 1, lenght + NUM_OF_FORECASTS)
    lower_series = pd.Series(conf[:, 0], index=forecast_index)
    upper_series = pd.Series(conf[:, 1], index=forecast_index)

    fig, ax = plt.subplots(1, figsize=(15, 4))

    ax.plot(data, label='Actual')
    ax.plot(predictions, label='Predictions')
    ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    ax.set_xticks(np.arange(lenght + NUM_OF_FORECASTS))

    ax.set_title('Predictions vs Actuals')
    ax.legend(loc="upper left")

    image_file = BytesIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)
    return image_file


def get_prediction_plot_image(tsa: TSADataset, model):
    key_name = 'prediction_plot'
    if not tsa.exist_image(key_name):
        tsa.put_image(create_prediction_plot_image(tsa.data_points, model), key_name)
    return tsa.url(key_name)


def get_actual_prediction_data_dict(data, model, actual_count, prediction_count):
    actual_prediction_data_dic = {}

    lenght = len(data)
    predictions = model.predict(lenght + 1, lenght + prediction_count)

    data_points = data[-actual_count:] + list(predictions)
    actual_prediction_data_dic["data"] = data_points
    actual_prediction_data_dic["indices"] = np.arange(lenght - actual_count, lenght + prediction_count)
    actual_prediction_data_dic["actual_count"] = actual_count
    actual_prediction_data_dic["prediction_count"] = prediction_count

    return actual_prediction_data_dic


def create_model_acf_plot_image(residuals, lag_number):
    fig, ax = plt.subplots(1, figsize=(15, 4))

    plot_acf(residuals, lags=lag_number if lag_number < len(residuals) else len(residuals) - 1, ax=ax)

    image_file = BytesIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)
    return image_file


def get_model_acf_plot_image(tsa: TSADataset, residuals, lag_number):
    key_name = 'model_acf_plot'
    if not tsa.exist_image(key_name):
        tsa.put_image(create_model_acf_plot_image(residuals, lag_number), key_name)
    return tsa.url(key_name)


def create_model_error_density_plot_image(residuals):
    fig, ax = plt.subplots(1, figsize=(15, 4))

    residuals.plot(
        ax=ax,
        kind='kde',
        title='ARIMA Fit Residual Error Density Plot',
        xlabel='Residual Values',
        grid=True,
    )

    image_file = BytesIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)
    return image_file


def get_model_error_density_plot_image(tsa: TSADataset, residuals):
    key_name = 'error_density_plot'
    if not tsa.exist_image(key_name):
        tsa.put_image(create_model_error_density_plot_image(residuals), key_name)
    return tsa.url(key_name)


def get_model_normality_test(residuals):
    test_res_dict = {}

    pvalues = acorr_ljungbox(residuals, lags=10)['lb_pvalue'].to_numpy()
    test_res_dict["pvalues"] = pvalues
    test_res_dict["independence"] = "Independent" if np.all(pvalues >= 0.05) else "Not Independent"

    return test_res_dict


def create_differencing_plot(data, i):
    fig, axs = plt.subplots(1, 2, figsize=(20, 2.5))

    ax = axs[0]
    lenght = len(data)
    ax.plot(np.arange(lenght), data,
            c=np.random.choice(['olive', 'hotpink', 'turquoise', 'firebrick', 'navy', 'goldenrod']))
    ax.set_xticks(np.arange(lenght))
    ax.set_xlabel("Index")
    ax.set_ylabel("Differenced Values")
    ax.set_title(f"Order {i} Differenced Data")

    plot_acf(data, lags=10, ax=axs[1])

    image_file = BytesIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)

    return image_file


def get_adf_test_output(data):
    output = f"Std of the differenced series: {np.std(data)}"

    result = adfuller(data)

    output += f'ADF Statistic: {result[0]}\n'
    output += f'p-value: {result[1]}\n'
    output += f'Critical Values:\n'
    for key, value in result[4].items():
        output += f'\t{key}: {value:.3f}\n'

    stationary = (result[1] <= 0.05) & (result[4]['5%'] > result[0])
    stationary = "Stationary" if stationary else "Non-stationary"

    return output, stationary


def get_stationarity_check_series(tsa: TSADataset):
    data = pd.Series(tsa.data_points)

    stationarity_plot_series = []
    stationarity_test_output_series = []
    stationarity_test_result_series = []

    for i in range(11):
        key_name = f'stationarity_plot_{i}'
        if not tsa.exist_image(key_name):
            tsa.put_image(create_differencing_plot(data, i), key_name)
        stationarity_plot_series.append(tsa.url(key_name))
        test_out, test_res = get_adf_test_output(data)
        stationarity_test_output_series.append(test_out)
        stationarity_test_result_series.append(test_res)
        if test_res == "Stationary":
            break

        data = data.diff().fillna(0)

    stationarity_checks = zip(stationarity_plot_series, stationarity_test_output_series,
                              stationarity_test_result_series)

    return stationarity_checks, data


def create_acf_pacf_plot_image(data, lag_number):
    fig, axs = plt.subplots(1, 2, figsize=(20, 4))

    length = len(data)
    plot_acf(data, lags=lag_number if lag_number < length else length - 1, ax=axs[0])
    plot_pacf(data, lags=lag_number if lag_number < length / 2 else length / 2 - 1, ax=axs[1], method="ywm")

    image_file = BytesIO()
    fig.savefig(image_file, format='svg')
    image_file.seek(0)

    return image_file


def get_acf_pacf_plot_image(tsa: TSADataset, data, lag_number):
    key_name = 'acf_pacf_plot'
    if not tsa.exist_image(key_name):
        tsa.put_image(create_acf_pacf_plot_image(data, lag_number), key_name)
    return tsa.url(key_name)
