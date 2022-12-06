from django.urls import path
from tsa.views import IndexView, VisualizationView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('visualization/', VisualizationView.as_view(), name='visualization'),
]