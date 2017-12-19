from django.db import models

# Create your models here.
class Member(models.Model):

	"""
	The core model of a member.
	"""

	member_name = models.CharField(max_length=30)
	graduate_year = models.IntegerField(default=0)
	description_text = models.TextField(max_length=500, default='')

	def __str__(self):
		return 'member_name: ' + self.member_name + ', graduate at: ' + str(self.graduate_year)



