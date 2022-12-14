from django.views.generic.edit import FormView
from django.views.generic.base import TemplateView
from tsa.forms import UploadFileForm
import pandas as pd
from tsa.serivces import get_data_plot_image, get_auto_arima_model, get_prediction_plot_image, get_start_end_points_dict, get_actual_prediction_data_dict, get_model_acf_plot_image, get_model_error_density_plot_image, get_model_normality_test, get_stationarity_check_series, get_acf_pacf_plot_image


class IndexView(FormView):
    template_name = 'tsa/index.html'
    form_class = UploadFileForm
    success_url = 'visualization/'

    def form_valid(self, form):
        df = pd.read_csv(self.request.FILES['file'], header=None)
        self.request.session['title'] = form.cleaned_data['title']
        self.request.session['data-points'] = df.iloc[:, 0].tolist()
        return super().form_valid(form)


class VisualizationView(TemplateView):
    PREVIEW_COUNT = 8
    template_name = "tsa/visualization.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = self.request.session.get('title', None)
        if not context['title']:
            return context
        
        data = pd.Series(self.request.session.get('data-points'))
        context.update(get_start_end_points_dict(data, self.PREVIEW_COUNT))
        context['summary'] = data.describe().to_dict()
        context['plot'] = get_data_plot_image(data)
        
        return context


class PreprocessingView(TemplateView):
    template_name = "tsa/preprocess.html"
    LAG_NUMBER = 20

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = self.request.session.get('title', None)
        if not context['title']:
            return context

        data = self.request.session.get('data-points')
        stationarity_checks, differenced_data = get_stationarity_check_series(data)
        context["stationarity_checks"] = stationarity_checks
        context["acf_pacf"] = get_acf_pacf_plot_image(differenced_data, self.LAG_NUMBER)
        return context


class TSAModelView(TemplateView):
    template_name = "tsa/model.html"
    ACTUAL_PREVIEW_COUNT = 8
    PREDICTION_PREVIEW_COUT = 5

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = self.request.session.get('title', None)
        if not context['title']:
            return context

        data = self.request.session.get('data-points')

        model, search_output = get_auto_arima_model(data)
        self.request.session['residuals'] = list(model.resid)
        context['search_output'] = search_output
        context['prediction_plot'] = get_prediction_plot_image(data, model)
        context.update(get_actual_prediction_data_dict(data, model, self.ACTUAL_PREVIEW_COUNT, self.PREDICTION_PREVIEW_COUT))
        return context


class ModelDiagnosticsView(TemplateView):
    template_name = "tsa/diagnostics.html"
    LAG_NUMBER = 20

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = self.request.session.get('title', None)
        if not context['title']:
            return context
            
        if 'residuals' not in self.request.session:
            return context

        residuals = pd.Series(self.request.session.get('residuals'))
        context["summary"] = residuals.describe().to_dict()
        context["acf_plot"] = get_model_acf_plot_image(residuals, self.LAG_NUMBER)
        context["error_density_plot"] = get_model_error_density_plot_image(residuals)
        context.update(get_model_normality_test(residuals))
        return context
