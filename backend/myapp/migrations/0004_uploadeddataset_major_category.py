from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0003_uploadedalgorithm_target_column_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadeddataset',
            name='major_category',
            field=models.CharField(
                choices=[
                    ('Environment', 'Environment'),
                    ('Finance & Economics', 'Finance & Economics'),
                    ('Geography', 'Geography'),
                    ('Industrial IoT', 'Industrial IoT'),
                    ('Mobility & Transport', 'Mobility & Transport'),
                    ('Systems & Monitoring', 'Systems & Monitoring'),
                    ('Others', 'Others'),
                ],
                default='Others',
                max_length=64,
            ),
        ),
    ]
