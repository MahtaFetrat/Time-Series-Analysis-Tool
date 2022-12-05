from django.views.generic.edit import FormView
from tsa.forms import UploadFileForm
import pandas as pd

# Create your views here.

class IndexView(FormView):
    template_name = 'tsa/index.html'
    form_class = UploadFileForm
    success_url = '/'

    def form_valid(self, form):
        df = pd.read_csv(self.request.FILES['file'])
        return super().form_valid(form)
