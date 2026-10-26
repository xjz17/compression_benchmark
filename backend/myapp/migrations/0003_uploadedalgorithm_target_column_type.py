from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_uploadedalgorithm'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedalgorithm',
            name='target_column_type',
            field=models.CharField(
                choices=[('int', 'Integer columns'), ('float', 'Float columns (and integer columns)')],
                default='float',
                max_length=8,
            ),
        ),
    ]
