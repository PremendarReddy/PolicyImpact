from django.urls import path
from . import views
from models.views import *

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/', views.chat, name='chat'),
    path('process/', views.chat_process, name='chat_process'),
    path('news/', views.live_news, name='live_news'),
    path('about',views.about,name="about")
]
