import json
import os

from django.conf import settings
from django.shortcuts import render, redirect


def main(request):
    #return redirect(to="app_main:confluence")
    return render(request, "app_main/index.html", context={
        "title": "AI-delix",
        "description": "AI-delix description"})


def report(request):
    return render(request, "app_main/report.html", context={
        "title": "AI-delix - Reports",
        "description": "Reports output",
    })

