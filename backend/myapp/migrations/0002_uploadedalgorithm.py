# Generated manually for user algorithm uploads

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedAlgorithm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('display_name', models.CharField(max_length=255)),
                ('storage_id', models.CharField(db_index=True, max_length=32)),
                ('original_filename', models.CharField(max_length=512)),
                ('implement_language', models.CharField(
                    choices=[('python', 'Python'), ('java', 'Java'), ('cpp', 'C++')],
                    max_length=16,
                )),
                ('lossy_lossless', models.CharField(
                    choices=[('lossless', 'Lossless'), ('lossy', 'Lossy')],
                    default='lossless',
                    max_length=16,
                )),
                ('entry_script', models.CharField(blank=True, default='', max_length=512)),
                ('processing_status', models.CharField(
                    choices=[
                        ('pending', 'Pending'),
                        ('running', 'Running'),
                        ('completed', 'Completed'),
                        ('failed', 'Failed'),
                    ],
                    db_index=True,
                    default='pending',
                    max_length=16,
                )),
                ('processing_error', models.TextField(blank=True, default='')),
                ('storage_path', models.CharField(max_length=1024)),
                ('created_at', models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ('owner', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='uploaded_algorithms',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddConstraint(
            model_name='uploadedalgorithm',
            constraint=models.UniqueConstraint(
                fields=('owner', 'storage_id'),
                name='uniq_owner_algo_storage_id',
            ),
        ),
    ]
