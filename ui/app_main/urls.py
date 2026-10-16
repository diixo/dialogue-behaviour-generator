
from django.urls import path
from . import views

app_name = "app_main"

urlpatterns = [
    path("", views.main, name="main"),
    path("report", views.report, name="report"),
    path("data/", views.datasets, name="datasets"),
    path("data/<str:split>/<str:episode_id>/", views.episode, name="episode"),
    path("experiments/", views.experiments, name="experiments"),
    path("experiments/<str:identifier>/", views.experiment, name="experiment"),
    path("benchmarks/", views.reports, name="reports"),
    path("benchmarks/compare/", views.compare, name="compare"),
    path("benchmarks/<str:identifier>/", views.report_detail, name="report_detail"),
    path("benchmarks/<str:identifier>/predictions/<int:position>/", views.prediction, name="prediction"),
]
