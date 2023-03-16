from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
        path('',views.predictor,name='predictor'),
        path('result',views.form_info,name='result')
]
