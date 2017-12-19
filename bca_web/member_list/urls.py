from django.conf.urls import url
from . import views

app_name = 'member_list'
urlpatterns = [
	url(r'^$', views.index, name='member_index')
]
