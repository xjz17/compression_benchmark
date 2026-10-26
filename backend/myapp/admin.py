from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

from .models import UploadedDataset, User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    ordering = ('email',)
    list_display = ('email', 'is_staff', 'is_active', 'date_joined')
    search_fields = ('email',)
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active'),
        }),
    )


@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display = (
        'display_name',
        'owner',
        'processing_status',
        'is_public',
        'file_format',
        'created_at',
    )
    list_filter = ('processing_status', 'is_public', 'file_format')
    search_fields = ('display_name', 'owner__email', 'storage_id')
    readonly_fields = ('created_at', 'storage_path', 'storage_id')
