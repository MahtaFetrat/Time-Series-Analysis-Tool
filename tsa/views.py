from django.views.generic.edit import FormView
from django.views.generic.base import TemplateView
from tsa.forms import UploadFileForm
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# Create your views here.

class IndexView(FormView):
    template_name = 'tsa/index.html'
    form_class = UploadFileForm
    success_url = 'visualization/'

    def form_valid(self, form):
        print(self.request.FILES['file'])
        df = pd.read_csv(self.request.FILES['file'])
        self.request.session['title'] = form.cleaned_data['title']
        self.request.session['data'] = df.iloc[:, 0].tolist()
        return super().form_valid(form)


class VisualizationView(TemplateView):
    # TODO: threading
    # TODO: separate service

    PREVIEW_COUNT = 8
    template_name = "tsa/visualization.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.set_context_datapoints(context)
        self.set_context_summary(context)
        self.set_context_plot(context)
        context['title'] = self.request.session['title']
        
        return context

    def set_context_datapoints(self, context):
        data = self.request.session.get('data')

        context['start_indices'] = range(1, self.PREVIEW_COUNT + 1)
        context['start_data'] = data[:self.PREVIEW_COUNT]
        context['end_indices'] = range(len(data) - self.PREVIEW_COUNT, len(data))
        context['end_data'] = data[-self.PREVIEW_COUNT:]
    
    def set_context_summary(self, context):
        data = pd.Series(self.request.session.get('data'))
        summary = data.describe().to_dict()
        summary = {k: round(v, 2) for k, v in summary.items()}
        context['summary'] = summary

    def set_context_plot(self, context):
        data = pd.Series(self.request.session.get('data'))

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
        context['plot'] = image_file.getvalue()


class TSAModelView(TemplateView):
    # TODO: threading
    # TODO: separate service

    template_name = "tsa/model.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        y = self.request.session.get('data')

        model = self.get_auto_arima_model(y)
        self.set_contex_prediction_plot(y, model, context)
        return context
        
    def get_auto_arima_model(self, y):
        auto_arima_model = auto_arima(
            y, 
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
        model = ARIMA(y, order=auto_arima_model.get_params()['order']).fit()
        # print("***********************\n")
        # print(model.get_forecast(1).conf_int(alpha=0.05))
        return model
        
    
    def set_contex_prediction_plot(self, y, model, context):
        NUM_OF_FORECASTS = 5
        
        lenght = len(y)
        predictions = model.predict(1, lenght + NUM_OF_FORECASTS - 1)

        conf = model.get_forecast(NUM_OF_FORECASTS - 1).conf_int(alpha=0.05)
        conf = np.insert(conf, 0, np.array([predictions[lenght - 1], predictions[lenght - 1]]), 0)

        forecast_index = np.arange(lenght - 1, lenght + NUM_OF_FORECASTS - 1)
        lower_series = pd.Series(conf[:, 0], index=forecast_index)
        upper_series = pd.Series(conf[:, 1], index=forecast_index)

        fig, ax = plt.subplots(1, figsize=(15,4))

        ax.plot(y, label='Actual')
        ax.plot(predictions, label='Predictions')
        ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
        ax.set_title('Predictions vs Actuals')
        ax.legend(loc="upper left")

        image_file = StringIO()
        fig.savefig(image_file, format='svg')
        image_file.seek(0)
        context['prediction_plot'] = image_file.getvalue()