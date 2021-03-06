from django.shortcuts import render, redirect, get_object_or_404
from django.views import generic
from django.contrib.auth.decorators import login_required
from . import models
from . import forms

# Create your views here.
def song_list(request, template_name='songs/song_list.html'):
	songs = models.Song.objects.all()
	_context = {}
	_context['object_list'] = songs
	return render(request, template_name, _context)

@login_required
def song_create(request, template_name='songs/song_form.html'):
	user = request.user
	if request.method == 'POST':
		# Parse form
		form = forms.SongForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
		return redirect('songs:index')
	else:
		form = forms.SongForm()
	_context = {'form': form}
	return render(request, template_name, _context)

@login_required
def song_update(request, pk, template_name='songs/song_form.html'):
	song = get_object_or_404(models.Song, pk=pk)
	if request.method == 'POST':

		form = forms.SongForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
		return redirect('songs:index')
	else:
		form = forms.SongForm()
	_context = {'form': form}
	return render(request, template_name, _context)

@login_required
def song_delete(request, pk, template_name='songs/song_confirm_delete.html'):
	song = get_object_or_404(models.Song, pk=pk)
	if request.method == 'POST':
		song.delete()
		return redirect('songs:index')
	_context = {'object': song}
	return render(request, template_name, _context)

