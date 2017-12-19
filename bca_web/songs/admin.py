from django.contrib import admin

# Register your models here.
from .models import Song

class SongAdmin(admin.ModelAdmin):
	fields = ['song_name', 'song_pdf']

admin.site.register(Song, SongAdmin)