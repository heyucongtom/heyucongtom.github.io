from django.conf.urls import url, include
from django.urls import path
from . import views

app_name = 'main_index'
urlpatterns = [
	url(r'^$', views.index, name='index'),
	path('accounts/', include('django.contrib.auth.urls'))
]