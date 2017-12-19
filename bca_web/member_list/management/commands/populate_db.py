from django.core.management.base import BaseCommand, CommandError
from member_list.models import Member
from django.conf import settings
import json
import os

OPT_GRAD_YEAR = 'graduate_year'
MEMBER_DATA_FILEPATH = 'member_list/static/member_list/members_2017_11.json'

class Command(BaseCommand):
	help = 'Populate db with initial members.'

	def add_arguments(self, parser):
		parser.add_argument(OPT_GRAD_YEAR, nargs='+', type=int)

	def handle(self, *args, **options):
		# Clean first.
		for member in Member.objects.all():
			member.delete()


		# Adding member from list
		with open(os.path.join(settings.BASE_DIR, MEMBER_DATA_FILEPATH)) as f:
			for year in options[OPT_GRAD_YEAR]:
				data = json.load(f)
				for member in data:
					
					if member['fields']['graduate_year'] not in options[OPT_GRAD_YEAR]:
						continue

					print("Inserting: %s" % member['fields']['member_name'])
					_member = Member(member_name=member['fields']['member_name'], graduate_year=member['fields']['graduate_year'])
					if 'description_text' in member['fields']:
						_member.description_text = member['fields']['description_text']
					_member.save()
				