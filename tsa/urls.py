from django.urls import path
from tsa.views import IndexView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
]