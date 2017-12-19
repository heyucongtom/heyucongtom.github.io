from django.db import models
from django.db.models.signals import post_delete

# Create your models here.
class Song(models.Model):

	def __str__(self):
		return self.song_name

	song_name = models.CharField(max_length=30, default='', unique=True)
	song_pdf = models.FileField(upload_to='songs/%Y', default='')
