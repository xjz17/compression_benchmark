"""Load saved user-upload benchmark artifacts for the Upload page."""

from __future__ import annotations

import logging
import os

import pandas as pd
from rest_framework import status
from rest_framework.response import Response

from .models import UploadedDataset
from .report_filters import (
    filter_compression_report_dataframe,
    parse_report_filter_params,
    rows_to_benchmark_items,
)
from .upload_serializers import UploadedDatasetSerializer
from .upload_storage import (
    resolve_user_upload_original_size,
    resolve_user_upload_report_csv,
    resolve_user_upload_statistic_csv,
    resolve_user_upload_tsfile_dir,
    user_upload_report_csv_path,
)

logger = logging.getLogger(__name__)


def build_user_upload_results_payload(
    record: UploadedDataset,
    query_params,
) -> dict | Response:
    data_type = (query_params.get('data_type') or record.data_type or '').strip()
    implement_language = (
        query_params.get('implement_language') or record.implement_language or ''
    ).strip()
    if data_type and implement_language:
        report_path = user_upload_report_csv_path(record, data_type, implement_language)
        if not os.path.isfile(report_path):
            report_path = None
    else:
        report_path = resolve_user_upload_report_csv(record)
    if not report_path:
        return Response(
            {
                'error': (
                    'No compression result file found for this upload. '
                    'Re-run the benchmark while signed in to regenerate artifacts.'
                ),
            },
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        output_report_df = pd.read_csv(report_path)
    except Exception as e:
        logger.exception('Failed to read upload report %s', report_path)
        return Response(
            {'error': f'Invalid compression result CSV: {e}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    original_size = resolve_user_upload_original_size(record, data_type=data_type or None)
    tsfile_dir = resolve_user_upload_tsfile_dir(record)
    filter_kw = parse_report_filter_params(query_params)
    filtered_df = filter_compression_report_dataframe(
        output_report_df,
        implement_language=implement_language or record.implement_language,
        **filter_kw,
    )
    items = rows_to_benchmark_items(
        filtered_df,
        original_size,
        tsfile_dir=tsfile_dir,
        process_id=record.storage_id,
    )
    items_all = rows_to_benchmark_items(
        output_report_df.dropna(subset=['algorithm']),
        original_size,
        tsfile_dir=tsfile_dir,
        process_id=record.storage_id,
    )

    statistic_data = []
    stat_path = resolve_user_upload_statistic_csv(record)
    if stat_path:
        try:
            statistic_data = pd.read_csv(stat_path).to_dict(orient='records')
        except Exception:
            logger.exception('Failed to read statistic CSV %s', stat_path)

    return {
        'upload': UploadedDatasetSerializer(record).data,
        'items': items,
        'items_all': items_all,
        'statistics': statistic_data,
        'charts': [],
        'process_id': record.storage_id,
    }
