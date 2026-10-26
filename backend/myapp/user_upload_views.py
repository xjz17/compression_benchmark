import mimetypes
import os

from django.http import FileResponse, Http404
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import UploadedDataset
from .upload_serializers import (
    UploadedDatasetSerializer,
    UploadedDatasetUpdateSerializer,
)
from .upload_storage import delete_user_upload, resolve_user_upload_chart_path
from .user_upload_results import build_user_upload_results_payload


class MyUploadsListView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        qs = UploadedDataset.objects.filter(owner=request.user).order_by('-created_at')
        serializer = UploadedDatasetSerializer(qs, many=True)
        return Response({'items': serializer.data}, status=status.HTTP_200_OK)


class MyUploadDetailView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def _get_owned(self, request, pk: int) -> UploadedDataset:
        return UploadedDataset.objects.get(pk=pk, owner=request.user)

    def patch(self, request, pk, *args, **kwargs):
        try:
            record = self._get_owned(request, pk)
        except UploadedDataset.DoesNotExist:
            return Response({'error': 'Upload not found.'}, status=status.HTTP_404_NOT_FOUND)

        serializer = UploadedDatasetUpdateSerializer(
            record,
            data=request.data,
            partial=True,
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(
            {'item': UploadedDatasetSerializer(record).data},
            status=status.HTTP_200_OK,
        )

    def delete(self, request, pk, *args, **kwargs):
        try:
            record = self._get_owned(request, pk)
        except UploadedDataset.DoesNotExist:
            return Response({'error': 'Upload not found.'}, status=status.HTTP_404_NOT_FOUND)

        delete_user_upload(record)
        return Response(status=status.HTTP_204_NO_CONTENT)


class MyUploadResultsView(APIView):
    """Return compression leaderboard, statistics, and chart URLs for a saved upload."""

    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        try:
            record = UploadedDataset.objects.get(pk=pk, owner=request.user)
        except UploadedDataset.DoesNotExist:
            return Response({'error': 'Upload not found.'}, status=status.HTTP_404_NOT_FOUND)

        payload = build_user_upload_results_payload(record, request.query_params)
        if isinstance(payload, Response):
            return payload
        return Response(payload, status=status.HTTP_200_OK)


class MyUploadChartView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, chart_key, *args, **kwargs):
        try:
            record = UploadedDataset.objects.get(pk=pk, owner=request.user)
        except UploadedDataset.DoesNotExist:
            raise Http404('Upload not found.')

        chart_path = resolve_user_upload_chart_path(record, chart_key)
        if not chart_path or not os.path.isfile(chart_path):
            raise Http404('Chart image not found')

        content_type, _ = mimetypes.guess_type(chart_path)
        if content_type is None:
            content_type = 'image/png'
        return FileResponse(open(chart_path, 'rb'), content_type=content_type)
