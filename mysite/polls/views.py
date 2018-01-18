from django.shortcuts import render
from django.shortcuts import get_object_or_404
from django.shortcuts import reverse
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.template import loader
from django.http import Http404
from django.views import generic
from django.utils import timezone
# Create your views here.
from .models import Choice, Question

# def index(request):
# 	"""
# 	displays the latest 5 poll questions in the system.
# 	"""
# 	latest_question_list = Question.objects.order_by('-pub_date')
# 	# Convention here, omitting /template
# 	template = loader.get_template("polls/index.html")

# 	context = {'latest_question_list': latest_question_list}
# 	# return HttpResponse(template.render(context, request))

# 	# Shortcut using render
# 	return render(request, "polls/index.html", context)
class IndexView(generic.ListView):
	template_name = 'polls/index.html' # Tell it to find correct template
	# Default is <app name>/<model_name>_list.html

	# Default is question_list
	context_object_name = 'latest_question_list'

	def get_queryset(self):
		return Question.objects.filter( 
			pub_date__lte=timezone.now()
		).order_by('-pub_date')[:5]

class DetailView(generic.DetailView):
	model = Question
	template_name = 'polls/detail.html'
	def get_queryset(self):
		return Question.objects.filter(pub_date__lte=timezone.now())

class ResultsView(generic.DetailView):
	model = Question
	template_name = 'polls/results.html'



# def detail(request, question_id):
# 	question = get_object_or_404(Question, pk=question_id)
# 	return render(request, "polls/detail.html", {'question': question})
	# return HttpResponse("You're looking at question %s." % question_id)

# def results(request, question_id):
# 	question = get_object_or_404(Question, pk=question_id)
# 	return render(request, "polls/results.html", {'question': question})

def vote(request, question_id):
	question = get_object_or_404(Question, pk=question_id)
	try:
		selected_choice = question.choice_set.get(pk=request.POST['choice'])
	except(KeyError, Choice.DoesNotExist):
		# Redisplay the question voting form
		return render(request, 'polls/detail.html', 
			{'question': question, 'error_message': "You didn't select a choice"})
	else:
		selected_choice.votes += 1
		selected_choice.save()
		# Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
		return HttpResponseRedirect(reverse('polls:results', args=(question.id, )))
