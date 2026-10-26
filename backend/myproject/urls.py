"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import re

from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include, re_path
from django.conf import settings
from django.views.static import serve as _static_serve


def _health(_request):
    """So ``GET /`` returns 200 — a bare 404 on ``/`` is often mistaken for a dead server."""
    return JsonResponse({"status": "ok", "service": "web_compression", "api": "/api/"})


urlpatterns = [
    path("", _health),
    path("admin/", admin.site.urls),
    path('api/', include('myapp.urls')),
]

# ``static()`` is a no-op when DEBUG is False. Use explicit ``serve`` routes so
# Admin CSS works under ``runserver`` without forcing DEBUG=True. Nginx should
# still alias ``/static/`` in production.
def _document_root_patterns(url_prefix: str, document_root: str):
    prefix = url_prefix.strip('/')
    return [
        re_path(
            rf'^{re.escape(prefix)}/(?P<path>.*)$',
            _static_serve,
            {'document_root': document_root},
        ),
    ]


urlpatterns += _document_root_patterns(settings.STATIC_URL, settings.STATIC_ROOT)
urlpatterns += _document_root_patterns(settings.MEDIA_URL, settings.MEDIA_ROOT)
