from django.views.generic.edit import FormView
from django.views.generic.base import TemplateView
from tsa.forms import UploadFileForm
import pandas as pd
from tsa.serivces import get_data_plot_image, get_auto_arima_model, get_prediction_plot_image, get_data_summary, get_start_end_points_dict


class IndexView(FormView):
    template_name = 'tsa/index.html'
    form_class = UploadFileForm
    success_url = 'visualization/'

    def form_valid(self, form):
        df = pd.read_csv(self.request.FILES['file'])
        self.request.session['title'] = form.cleaned_data['title']
        self.request.session['data'] = df.iloc[:, 0].tolist()
        return super().form_valid(form)


class VisualizationView(TemplateView):
    PREVIEW_COUNT = 8
    template_name = "tsa/visualization.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        data = pd.Series(self.request.session.get('data'))
        context.update(get_start_end_points_dict(data, self.PREVIEW_COUNT))
        context['summary'] = get_data_summary(data)
        context['plot'] = get_data_plot_image(data)
        context['title'] = self.request.session['title']
        
        return context


class TSAModelView(TemplateView):
    template_name = "tsa/model.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        data = self.request.session.get('data')

        model, search_output = get_auto_arima_model(data)
        context['search_output'] = search_output
        context['prediction_plot'] = get_prediction_plot_image(data, model)
        return context
