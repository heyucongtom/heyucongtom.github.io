from django.shortcuts import render

# Create your views here.
def index(request):
	_context = {}
	return render(request, "main_index/index.html", _context)