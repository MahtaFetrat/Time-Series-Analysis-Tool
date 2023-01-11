from django.contrib.postgres import fields
from django.db import models

from storages.backends.s3boto3 import S3Boto3Storage


class MediaStorage(S3Boto3Storage):
    location = "pics"


class TSADataset(models.Model):
    title = models.CharField(max_length=32)
    data_points = fields.ArrayField(models.FloatField())
    residuals = fields.ArrayField(models.FloatField(), null=True)
    media_storage = MediaStorage()

    def storage_key(self, name):
        return f"{self.id}-{name}.svg"

    def url(self, name):
        return self.media_storage.url(self.storage_key(name)).replace('minioserver', 'localhost')

    def exist_image(self, name):
        return self.media_storage.exists(self.storage_key(name))

    def put_image(self, data, name):
        self.media_storage.save(self.storage_key(name), data)
