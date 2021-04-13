from django.db import models

# Create your models here.
class register(models.Model):
    username=models.CharField(max_length=50,null=True)
    password = models.CharField(max_length=50)
    email=models.CharField(max_length=50,null=True)

    