from django.shortcuts import render
from django.contrib.auth import login, authenticate
from django.shortcuts import redirect, render
from . import forms

# Create your views here.
def index(request):
	_context = {}
	return render(request, "main_index/index.html", _context)

def signup(request):
	if request.method == "POST":
		form = forms.SignupForm(request.POST)
		if form.is_valid():
			user = form.save() # Register the user.
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			# user = authenticate(username=username, password=password)
			login(request, user)
			return redirect('main_index:index')
	else:
		form = forms.SignupForm()
	return render(request, "registration/signup.html", { 'form': form })

