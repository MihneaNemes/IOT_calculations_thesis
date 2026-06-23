from flask import Flask, render_template, request, jsonify
import boto3
from datetime import datetime
import json

app = Flask(__name__)

# Initialize AWS DynamoDB client
dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

table_name = "BodyFatPredictions"
bucket_name = "bodyfat-predictor-images"
table = dynamodb.Table(table_name)

@app.route('/')
def index():
    try:
        # Scan DynamoDB table to get all predictions
        response = table.scan()
        items = response.get('Items', [])
        
        # Extract unique names
        names = sorted(set(item.get('name', 'N/A') for item in items if item.get('name')))
        
        # Group predictions by name
        predictions_by_name = {}
        for item in items:
            name = item.get('name', 'N/A')
            if name not in predictions_by_name:
                predictions_by_name[name] = []
            
            # Format Timestamp nicely
            raw_time = item.get('timestamp', '')
            try:
                dt_obj = datetime.fromisoformat(raw_time)
                formatted_time = dt_obj.strftime("%b %d, %Y - %H:%M")
            except:
                formatted_time = raw_time

            predictions_by_name[name].append({
                'id': item.get('prediction_id'), # Keep internal ID for fetch calls
                'name': name,
                'height_cm': float(item['height_cm']),
                'weight_kg': float(item['weight_kg']),
                'predicted_body_fat': float(item['predicted_body_fat']),
                'category': item['category'],
                'timestamp': formatted_time,
                'raw_timestamp': raw_time, # Keep raw time for sorting
                'has_images': bool(item.get('front_image_key') and item.get('side_image_key')),
                'front_key': item.get('front_image_key'),
                'side_key': item.get('side_image_key')
            })
        
        # Sort predictions by timestamp for each name
        for name in predictions_by_name:
            predictions_by_name[name].sort(key=lambda x: x['raw_timestamp'])
        
        # Prepare chart data
        chart_data = {}
        for name in predictions_by_name:
            timestamps = [p['timestamp'] for p in predictions_by_name[name]]
            body_fats = [p['predicted_body_fat'] for p in predictions_by_name[name]]
            chart_data[name] = {
                'labels': timestamps,
                'data': body_fats
            }
        
        return render_template('index.html', names=names, predictions_by_name=predictions_by_name, chart_data=json.dumps(chart_data))
    except Exception as e:
        return f"Error loading predictions: {str(e)}", 500

@app.route('/get_image_urls', methods=['POST'])
def get_image_urls():
    """Generates temporary presigned URLs for viewing images"""
    data = request.json
    front_key = data.get('front_key')
    side_key = data.get('side_key')

    urls = {}
    try:
        if front_key:
            urls['front'] = s3_client.generate_presigned_url('get_object',
                                                            Params={'Bucket': bucket_name, 'Key': front_key},
                                                            ExpiresIn=300) # 5 min expiry
        if side_key:
            urls['side'] = s3_client.generate_presigned_url('get_object',
                                                            Params={'Bucket': bucket_name, 'Key': side_key},
                                                            ExpiresIn=300)
        return jsonify(urls)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)