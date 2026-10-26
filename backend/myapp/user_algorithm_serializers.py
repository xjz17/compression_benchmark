from rest_framework import serializers

from .models import UploadedAlgorithm


class UploadedAlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedAlgorithm
        fields = (
            'id',
            'display_name',
            'storage_id',
            'original_filename',
            'implement_language',
            'lossy_lossless',
            'target_column_type',
            'entry_script',
            'processing_status',
            'processing_error',
            'created_at',
        )
        read_only_fields = fields


class UploadedAlgorithmCreateSerializer(serializers.Serializer):
    display_name = serializers.CharField(max_length=255)
    implement_language = serializers.ChoiceField(
        choices=UploadedAlgorithm.LANGUAGE_CHOICES,
    )
    lossy_lossless = serializers.ChoiceField(
        choices=UploadedAlgorithm.MODE_CHOICES,
        default=UploadedAlgorithm.MODE_LOSSLESS,
    )
    target_column_type = serializers.ChoiceField(
        choices=UploadedAlgorithm.TARGET_COLUMN_CHOICES,
        default=UploadedAlgorithm.TARGET_FLOAT,
    )
