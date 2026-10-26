from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .auth_views import LoginView, LogoutView, MeView, RegisterView
from .user_algorithm_views import MyAlgorithmDetailView, MyAlgorithmsListView
from .user_upload_views import (
    MyUploadChartView,
    MyUploadDetailView,
    MyUploadResultsView,
    MyUploadsListView,
)
from .views import (
    UploadView,
    StaticDataView,
    StaticDataManifestView,
    TsFileStaticDataView,
    TsFileStaticArtifactDownloadView,
    StaticChartImageView,
    DatasetCatalogView,
    MethodCatalogView,
    EncodingOperatorCatalogView,
    DatasetColumnInfoView,
    DatasetCsvDownloadView,
    DatasetTsFileDownloadView,
    DatasetCatalogStatisticJsonView,
    DatasetCatalogCorrelationHeatmapsView,
    DatasetCatalogOperatorStepEntropyView,
    DatasetCatalogColumnDistributionChartView,
    DatasetCatalogAssetFileView,
    BenchmarkBoxplotManifestView,
)

router = DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('auth/register/', RegisterView.as_view(), name='auth-register'),
    path('auth/login/', LoginView.as_view(), name='auth-login'),
    path('auth/logout/', LogoutView.as_view(), name='auth-logout'),
    path('auth/me/', MeView.as_view(), name='auth-me'),
    path('upload/', UploadView.as_view(), name='upload'),
    path('my-algorithms/', MyAlgorithmsListView.as_view(), name='my-algorithms-list'),
    path('my-algorithms/<int:pk>/', MyAlgorithmDetailView.as_view(), name='my-algorithms-detail'),
    path('my-uploads/', MyUploadsListView.as_view(), name='my-uploads-list'),
    path('my-uploads/<int:pk>/', MyUploadDetailView.as_view(), name='my-uploads-detail'),
    path('my-uploads/<int:pk>/results/', MyUploadResultsView.as_view(), name='my-uploads-results'),
    path(
        'my-uploads/<int:pk>/charts/<str:chart_key>/',
        MyUploadChartView.as_view(),
        name='my-uploads-chart',
    ),
    path('staticdata/', StaticDataView.as_view(), name='staticdata'),
    path('staticdata/manifest/', StaticDataManifestView.as_view(), name='staticdata-manifest'),
    path('tsfile-staticdata/', TsFileStaticDataView.as_view(), name='tsfile-staticdata'),
    path(
        'tsfile-staticdata/download/',
        TsFileStaticArtifactDownloadView.as_view(),
        name='tsfile-staticdownload',
    ),
    path('datasets/', DatasetCatalogView.as_view(), name='datasets'),
    path('methods/', MethodCatalogView.as_view(), name='methods'),
    path('encoding-operators/', EncodingOperatorCatalogView.as_view(), name='encoding-operators'),
    path('datasets/<str:dataset_key>/columns/', DatasetColumnInfoView.as_view(), name='datasets-columns'),
    path('datasets/download/<str:dataset_key>/csv/', DatasetCsvDownloadView.as_view(), name='datasets-download-csv'),
    path('datasets/download/<str:dataset_key>/tsfile/', DatasetTsFileDownloadView.as_view(), name='datasets-download-tsfile'),
    path(
        'datasets/catalog-assets/<str:dataset_key>/statistics-json/',
        DatasetCatalogStatisticJsonView.as_view(),
        name='datasets-catalog-statistics-json',
    ),
    path(
        'datasets/catalog-assets/<str:dataset_key>/correlation-heatmaps/',
        DatasetCatalogCorrelationHeatmapsView.as_view(),
        name='datasets-catalog-correlation-heatmaps',
    ),
    path(
        'datasets/catalog-assets/<str:dataset_key>/operator-step-entropy/',
        DatasetCatalogOperatorStepEntropyView.as_view(),
        name='datasets-catalog-operator-step-entropy',
    ),
    path(
        'datasets/catalog-assets/<str:dataset_key>/column-charts/<str:filename>',
        DatasetCatalogColumnDistributionChartView.as_view(),
        name='datasets-catalog-column-distribution-chart',
    ),
    path(
        'datasets/catalog-assets/<str:dataset_key>/<str:filename>',
        DatasetCatalogAssetFileView.as_view(),
        name='datasets-catalog-asset-file',
    ),
    path('static-chart/<path:chart_path>/', StaticChartImageView.as_view(), name='static-chart-image'),
    path(
        'benchmark-boxplots-manifest/',
        BenchmarkBoxplotManifestView.as_view(),
        name='benchmark-boxplot-manifest',
    ),
]
