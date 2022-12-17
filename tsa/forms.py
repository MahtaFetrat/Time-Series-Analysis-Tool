from django import forms
import pandas as pd
from django.core.exceptions import ValidationError
from pandas.api.types import is_numeric_dtype

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()

    def clean(self):
        cleaned_data = super(UploadFileForm, self).clean()
        csv_file = cleaned_data.get('file')
        df = pd.read_csv(csv_file, header=None)
        csv_file.seek(0)

        if df.shape[1] != 1 or not is_numeric_dtype(df.iloc[:,0]):
            raise ValidationError("Sorry the file must be a single-column numeric csv!")
        
        return cleaned_data