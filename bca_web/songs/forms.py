from django.forms import ModelForm
from . import models

class SongForm(ModelForm):
	
	class Meta:
		model = models.Song
		fields = ['song_name', 'song_pdf']