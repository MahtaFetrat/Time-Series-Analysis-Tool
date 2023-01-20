from django.urls import path
from tsa.views import IndexView, VisualizationView, PreprocessingView, TSAModelView, ModelDiagnosticsView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('visualization/<int:tsa_id>', VisualizationView.as_view(), name='visualization'),
    path('preprocess/<int:tsa_id>', PreprocessingView.as_view(), name='preprocess'),
    path('model/<int:tsa_id>', TSAModelView.as_view(), name='tsa_model'),
    path('diagnostics/<int:tsa_id>', ModelDiagnosticsView.as_view(), name='model_diagnostics'),
]