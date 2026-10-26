from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

from .dataset_categories import (
    DATASET_MAJOR_CATEGORY_CHOICES,
    DATASET_MAJOR_CATEGORY_OTHERS,
)


class UserManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        if not email:
            raise ValueError('Email is required')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        return self._create_user(email, password, **extra_fields)


class User(AbstractUser):
    """Email-based login; no username."""

    username = None
    email = models.EmailField('email address', unique=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS: list[str] = []

    class Meta:
        db_table = 'myapp_user'


class UploadedDataset(models.Model):
    """Persisted user upload (benchmark artifacts under media/user_datasets/)."""

    PROCESSING_PENDING = 'pending'
    PROCESSING_RUNNING = 'running'
    PROCESSING_COMPLETED = 'completed'
    PROCESSING_FAILED = 'failed'
    PROCESSING_CHOICES = [
        (PROCESSING_PENDING, 'Pending'),
        (PROCESSING_RUNNING, 'Running'),
        (PROCESSING_COMPLETED, 'Completed'),
        (PROCESSING_FAILED, 'Failed'),
    ]

    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='uploaded_datasets',
    )
    display_name = models.CharField(max_length=255)
    storage_id = models.CharField(max_length=32, db_index=True)
    original_filename = models.CharField(max_length=512)
    file_format = models.CharField(max_length=16)
    data_type = models.CharField(max_length=8, blank=True, default='')
    implement_language = models.CharField(max_length=16, blank=True, default='')
    processing_status = models.CharField(
        max_length=16,
        choices=PROCESSING_CHOICES,
        default=PROCESSING_PENDING,
        db_index=True,
    )
    processing_error = models.TextField(blank=True, default='')
    is_public = models.BooleanField(default=False)
    major_category = models.CharField(
        max_length=64,
        choices=DATASET_MAJOR_CATEGORY_CHOICES,
        default=DATASET_MAJOR_CATEGORY_OTHERS,
    )
    storage_path = models.CharField(max_length=1024)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['owner', 'storage_id'],
                name='uniq_owner_storage_id',
            ),
        ]

    def __str__(self) -> str:
        vis = 'public' if self.is_public else 'private'
        return f'{self.display_name} ({self.owner.email}, {vis})'


class UploadedAlgorithm(models.Model):
    """User-submitted compressors evaluated against the owner's uploaded datasets."""

    PROCESSING_PENDING = 'pending'
    PROCESSING_RUNNING = 'running'
    PROCESSING_COMPLETED = 'completed'
    PROCESSING_FAILED = 'failed'
    PROCESSING_CHOICES = [
        (PROCESSING_PENDING, 'Pending'),
        (PROCESSING_RUNNING, 'Running'),
        (PROCESSING_COMPLETED, 'Completed'),
        (PROCESSING_FAILED, 'Failed'),
    ]

    LANGUAGE_PYTHON = 'python'
    LANGUAGE_JAVA = 'java'
    LANGUAGE_CPP = 'cpp'
    LANGUAGE_CHOICES = [
        (LANGUAGE_PYTHON, 'Python'),
        (LANGUAGE_JAVA, 'Java'),
        (LANGUAGE_CPP, 'C++'),
    ]

    MODE_LOSSLESS = 'lossless'
    MODE_LOSSY = 'lossy'
    MODE_CHOICES = [
        (MODE_LOSSLESS, 'Lossless'),
        (MODE_LOSSY, 'Lossy'),
    ]

    TARGET_INT = 'int'
    TARGET_FLOAT = 'float'
    TARGET_COLUMN_CHOICES = [
        (TARGET_INT, 'Integer columns only'),
        (TARGET_FLOAT, 'Float columns (and integer columns)'),
    ]

    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='uploaded_algorithms',
    )
    display_name = models.CharField(max_length=255)
    storage_id = models.CharField(max_length=32, db_index=True)
    original_filename = models.CharField(max_length=512)
    implement_language = models.CharField(max_length=16, choices=LANGUAGE_CHOICES)
    lossy_lossless = models.CharField(max_length=16, choices=MODE_CHOICES, default=MODE_LOSSLESS)
    target_column_type = models.CharField(
        max_length=8,
        choices=TARGET_COLUMN_CHOICES,
        default=TARGET_FLOAT,
    )
    entry_script = models.CharField(max_length=512, blank=True, default='')
    processing_status = models.CharField(
        max_length=16,
        choices=PROCESSING_CHOICES,
        default=PROCESSING_PENDING,
        db_index=True,
    )
    processing_error = models.TextField(blank=True, default='')
    storage_path = models.CharField(max_length=1024)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['owner', 'storage_id'],
                name='uniq_owner_algo_storage_id',
            ),
        ]

    def __str__(self) -> str:
        return f'{self.display_name} ({self.implement_language}, {self.lossy_lossless})'
