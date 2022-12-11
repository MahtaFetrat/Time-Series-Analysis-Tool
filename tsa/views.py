from django.views.generic.edit import FormView
from django.views.generic.base import TemplateView
from tsa.forms import UploadFileForm
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

# Create your views here.

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
