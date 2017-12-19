# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-11-27 23:04
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('songs', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='song',
            name='song_path',
            field=models.CharField(default='', max_length=255),
        ),
        migrations.AlterField(
            model_name='song',
            name='song_name',
            field=models.CharField(default='', max_length=30),
        ),
    ]
