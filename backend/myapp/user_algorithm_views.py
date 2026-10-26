import os
import tempfile

from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import UploadedAlgorithm
from .user_algorithm_pipeline import start_user_algorithm_pipeline
from .user_algorithm_serializers import (
    UploadedAlgorithmCreateSerializer,
    UploadedAlgorithmSerializer,
)
from .user_algorithm_storage import create_user_algorithm, delete_user_algorithm


class MyAlgorithmsListView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        qs = UploadedAlgorithm.objects.filter(owner=request.user).order_by('-created_at')
        return Response(
            {'items': UploadedAlgorithmSerializer(qs, many=True).data},
            status=status.HTTP_200_OK,
        )

    def post(self, request, *args, **kwargs):
        file = request.data.get('file')
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        meta = UploadedAlgorithmCreateSerializer(data=request.data)
        meta.is_valid(raise_exception=True)
        display_name = meta.validated_data['display_name'].strip()
        if not display_name:
            return Response({'error': 'Algorithm name is required'}, status=status.HTTP_400_BAD_REQUEST)

        lower = file.name.lower()
        if not (lower.endswith('.zip') or lower.endswith('.py') or lower.endswith('.java') or lower.endswith('.cpp')):
            return Response(
                {'error': 'Upload a .zip archive or a single .py / .java / .cpp source file.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if file.size > 50 * 1024 * 1024:
            return Response({'error': 'File size cannot exceed 50MB'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with tempfile.TemporaryDirectory(prefix='algo_upload_') as temp_dir:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, 'wb+') as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)
                record = create_user_algorithm(
                    user=request.user,
                    display_name=display_name,
                    implement_language=meta.validated_data['implement_language'],
                    lossy_lossless=meta.validated_data['lossy_lossless'],
                    target_column_type=meta.validated_data['target_column_type'],
                    uploaded_path=temp_path,
                    original_filename=file.name,
                )
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            return Response({'error': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        start_user_algorithm_pipeline(record.pk)
        return Response(
            {
                'item': UploadedAlgorithmSerializer(record).data,
                'message': 'Algorithm saved. Benchmarks running in background.',
            },
            status=status.HTTP_202_ACCEPTED,
        )


class MyAlgorithmDetailView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, pk, *args, **kwargs):
        try:
            record = UploadedAlgorithm.objects.get(pk=pk, owner=request.user)
        except UploadedAlgorithm.DoesNotExist:
            return Response({'error': 'Algorithm not found.'}, status=status.HTTP_404_NOT_FOUND)
        delete_user_algorithm(record)
        return Response(status=status.HTTP_204_NO_CONTENT)
