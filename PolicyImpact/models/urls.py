from django.urls import path
from . import views 
# from models.views import home

urlpatterns = [
    path('', views.home, name='home'),
    path('sentiment_analysis/', views.sentimentanalysis, name='sentimentanalysis'),
    path('economic_impact/', views.economicimpact, name='economicimpact'),
    path('forecasting/', views.forecasting, name='forecasting'),
    path('policy_optimization/', views.policyoptimization, name='policyoptimization'),
] 