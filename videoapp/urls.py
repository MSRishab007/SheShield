# videoapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.render_home, name='home'),
    path('gender/', views.video_upload_view, name='video_upload'),
    path('map/', views.show_map, name='show_map'),
    path('gesture/', views.gesture_view, name='gesture'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('gender/gender./', views.video_upload_view, name='video_upload'),
    path('gender/map/', views.show_map, name='show_map'),
    path('gender/gesture/', views.gesture_view, name='gesture'),
    path('gender/dashboard/', views.dashboard_view, name='dashboard'),
    path('gender/', views.video_upload_view, name='video_upload'),
    path('map/', views.show_map, name='show_map'),
    path('gesture/', views.gesture_view, name='gesture'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('contact/', views.contact_us_view, name='contact_us'),
    
]
