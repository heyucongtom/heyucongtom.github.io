from django.contrib import admin
from .models import Question, Choice, Voter


class ChoiceInline(admin.TabularInline):
	model = Choice
	extra = 3

class VoterInline(admin.TabularInline):
	model = Voter
	extra = 3
	
# Register your models here.
class QuestionAdmin(admin.ModelAdmin):
	list_display = ('question_text', 'pub_date', 'was_published_recently')
	fieldsets = [
		(None, {'fields': ['question_text']}),
		('Date information', {'fields':['pub_date'], 'classes':['collapse']}),
	]
	inlines = [ChoiceInline, VoterInline]
	list_filter = ['pub_date']
	search_fields = ['question_text']



admin.site.register(Question, QuestionAdmin)
