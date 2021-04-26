from django.shortcuts import render
from recognition import pipeline
from PIL import Image,ImageOps
from recognition.Face_Recogniton_Model import mtcnn as mt
from recognition.Face_Recogniton_Model import inception_resnet_v1
import requests
import urllib.request
import torch



def info(request):
    return render(request,"info.html")

def base(request):
    return render(request,"base.html")

def contactus(request):
    return render(request,"contactus.html")

def upload_image(request):
    context = {
        "image": "static/image/empty-image.png",
    }
    return render(request,"uploadimage.html",context=context)


def recognition(request):

    img_src = request.POST.get("img-src")
    try:
        image_file = request.FILES.getlist('file')[0]
        image = Image.open(image_file)
        fixed_image = ImageOps.exif_transpose(image)
        name = pipeline.face_detection_and_recognition(fixed_image)
    except:
        img_src = "static/image/empty-image.png"
        name = pipeline.face_detection_and_recognition()
    if name not in ["Face Not Detected","Not Match Found","Access Denied"]:
        context = {
            "image" : img_src,
        }
        return render(request,"Access_granted.html",context=context)

    else:
        context = {
            "image": img_src,
            "access":name
        }
        return render(request,"access_denied.html",context=context)

