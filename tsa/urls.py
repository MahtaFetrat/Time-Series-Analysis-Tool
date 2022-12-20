from django.urls import path
from tsa.views import IndexView, VisualizationView, TSAModelView, ModelDiagnosticsView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('visualization/', VisualizationView.as_view(), name='visualization'),
    path('model/', TSAModelView.as_view(), name='tsa_model'),
    path('diagnostics/', ModelDiagnosticsView.as_view(), name='model_diagnostics'),
]