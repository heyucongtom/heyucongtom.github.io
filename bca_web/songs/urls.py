from django.conf.urls import url
from . import views

app_name = 'songs'
urlpatterns = [
	url(r'^$', views.song_list, name='index'),
	url(r'^new$', views.song_create, name='new'),
	url(r'^edit/(?P<pk>\d+)$', views.song_update, name='edit'),
	url(r'^delete/(?P<pk>\d+)$', views.song_delete, name='delete')
	# url(r'^upload/success/', views.upload_success, name='upload_success'),
	# url(r'^upload', views.upload_song, name='upload')
	
]
