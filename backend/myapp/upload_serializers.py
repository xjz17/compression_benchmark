from rest_framework import serializers

from .models import UploadedDataset


class UploadedDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDataset
        fields = (
            'id',
            'display_name',
            'storage_id',
            'original_filename',
            'file_format',
            'data_type',
            'implement_language',
            'processing_status',
            'processing_error',
            'is_public',
            'major_category',
            'created_at',
        )
        read_only_fields = fields


class UploadedDatasetUpdateSerializer(serializers.ModelSerializer):
    display_name = serializers.CharField(max_length=255, required=False, allow_blank=False)
    is_public = serializers.BooleanField(required=False)

    class Meta:
        model = UploadedDataset
        fields = ('display_name', 'is_public')

    def validate_display_name(self, value):
        text = (value or '').strip()
        if not text:
            raise serializers.ValidationError('Display name cannot be empty.')
        return text
