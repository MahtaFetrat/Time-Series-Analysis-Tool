from django.views.generic.edit import FormView
from django.views.generic.base import TemplateView
from tsa.forms import UploadFileForm
import pandas as pd

# Create your views here.

class IndexView(FormView):
    template_name = 'tsa/index.html'
    form_class = UploadFileForm
    success_url = 'visualization/'

    def form_valid(self, form):
        df = pd.read_csv(self.request.FILES['file'])
        self.request.session['df'] = df.iloc[:, 0].tolist()
        return super().form_valid(form)


class VisualizationView(TemplateView):
    template_name = "tsa/visualization.html"
    
