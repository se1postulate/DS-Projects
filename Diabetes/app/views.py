from django.shortcuts import render
from django.http import HttpResponse
import pickle
import os
from .models import register

modulePath = os.path.dirname(__file__)
filePath = os.path.join(modulePath, 'Code/model.sav')


with open(filePath, 'rb') as f:
    model = pickle.load(f)

def  index(request):
        if  register.objects.filter(username=(request.POST.get('username'))).exists() and register.objects.filter(password=(request.POST.get('password'))).exists():
                        return render(request,"index.html")
        else:
                return render(request,'login.html')
def home(request):
        return render(request,'index.html')
def  predict(request):
    f=request.POST.get("father")
    m=request.POST.get("mother")
    fh=request.POST.get("fatherh")
    mh=request.POST.get("motherh")
    data=[f,m,fh,mh]
    preds = model.predict([data])
    if preds==1:
            result='More Chances Of Daibetes'
    else:
            result='less Chances Of Daibetes '
            
    #return HttpResponse(result)
    name=request.POST.get('name')
    age=request.POST.get('age')
    gender=request.POST.get('gender')
    return render(request, 'predict.html',{'result':result,'name':name,'age':age,'gender':gender})

def login(request):
        return render(request,'login.html')

def signup(request):
        un=request.POST.get('username')
        pa=request.POST.get('password')
        em=request.POST.get('email')
        data=register(username=un,password=pa,email=em)
        data.save()
        return render(request,'login.html')