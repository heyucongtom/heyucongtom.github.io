# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-11-16 06:44
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('member_list', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='member',
            name='description_text',
            field=models.TextField(default='', max_length=500),
        ),
        migrations.AddField(
            model_name='member',
            name='graduate_year',
            field=models.IntegerField(default=0),
        ),
    ]
