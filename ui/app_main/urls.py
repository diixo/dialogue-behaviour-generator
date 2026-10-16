
from django.urls import path
from . import views
from . import job_views
from . import chat_views

app_name = "app_main"

urlpatterns = [
    path('chat/', chat_views.chats, name='chats'),
    path('chat/<uuid:identifier>/', chat_views.chat_detail, name='chat'),
    path('chat/<uuid:identifier>/send/', chat_views.send_message, name='chat_send'),
    path('chat/<uuid:identifier>/status/', chat_views.chat_status, name='chat_status'),
    path('chat/<uuid:identifier>/stop/', chat_views.stop_chat, name='chat_stop'),
    path('jobs/', job_views.job_list, name='jobs'),
    path('jobs/train/', job_views.new_training, name='new_training'),
    path('jobs/benchmark/', job_views.new_benchmark, name='new_benchmark'),
    path('jobs/resume/', job_views.resume_training, name='resume_training'),
    path('jobs/<uuid:identifier>/', job_views.job_detail, name='job'),
    path('jobs/<uuid:identifier>/status/', job_views.job_status, name='job_status'),
    path('jobs/<uuid:identifier>/stop/', job_views.stop_job, name='stop_job'),
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
