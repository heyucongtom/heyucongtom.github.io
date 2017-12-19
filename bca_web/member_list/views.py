from django.http import HttpResponse
from django.shortcuts import render

from .models import Member

# Create your views here.
def index(request):
	"""
	Note: View only returns a response.
	"""
	_members = Member.objects.order_by('member_name')
	_context = {
		'members': _members
	}
	return render(request, "member_list/index.html", _context)

	