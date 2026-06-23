import boto3
from io import BytesIO
from decimal import Decimal
from datetime import datetime


class AWSManager:
    
    def __init__(self, bucket_name="bodyfat-predictor-mihnea-2026", table_name="BodyFatPredictions"):
        self.bucket_name = bucket_name
        self.table_name = table_name
        
        try:
            self.s3_client = boto3.client('s3')
            self.dynamodb = boto3.resource('dynamodb')
            self.table = self.dynamodb.Table(self.table_name)
        except Exception as e:
            print(f"AWS Error: {e}")
            self.s3_client = None
            self.dynamodb = None
            self.table = None
    
    def upload_images(self, prediction_id, front_image, side_image):

        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        try:
            front_key = f"images/{prediction_id}_front.jpg"
            side_key = f"images/{prediction_id}_side.jpg"
            
            front_buffer = BytesIO()
            side_buffer = BytesIO()
            front_image.save(front_buffer, format="JPEG")
            side_image.save(side_buffer, format="JPEG")
            front_buffer.seek(0)
            side_buffer.seek(0)
            
            self.s3_client.upload_fileobj(front_buffer, self.bucket_name, front_key)
            self.s3_client.upload_fileobj(side_buffer, self.bucket_name, side_key)
            
            return front_key, side_key
        except Exception as e:
            print(f"Error uploading images: {e}")
            raise
    
    def save_prediction(self, prediction_id, name, sex, height_cm, weight_kg, 
                       body_fat_percent, ffm, fat_mass, category, front_image_key, side_image_key):
        if not self.table:
            raise RuntimeError("DynamoDB table not initialized")
        
        try:
            self.table.put_item(
                Item={
                    'prediction_id': prediction_id,
                    'name': name,
                    'sex': sex,
                    'height_cm': Decimal(str(height_cm)),
                    'weight_kg': Decimal(str(weight_kg)),
                    'predicted_body_fat': Decimal(str(round(body_fat_percent, 2))),
                    'fat_free_mass_kg': Decimal(str(round(ffm, 2))),    
                    'fat_mass_kg': Decimal(str(round(fat_mass, 2))),    
                    'category': category,
                    'front_image_key': front_image_key,
                    'side_image_key': side_image_key,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            print(f"Error saving to DynamoDB: {e}")
            raise
