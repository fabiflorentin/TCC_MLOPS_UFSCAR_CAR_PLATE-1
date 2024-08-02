from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import boto3
import io
import cv2
import numpy as np
from PIL import Image

from fastapi.responses import StreamingResponse

# Inicialize o aplicativo FastAPI
app = FastAPI()

# Inicialize o cliente S3
s3_client = boto3.client('s3')

YOLOV8_DIR = '/tmp'

# Set or modify the value of YOLO_CONFIG_DIR
os.environ['YOLO_CONFIG_DIR'] = "/tmp"
# Verify that the value has been updated
print(f"YOLO_CONFIG_DIR: {os.getenv('YOLO_CONFIG_DIR')}")

from ultralytics import YOLO


# Defina o caminho do modelo YOLOv8
MODEL_PATH = '/tmp/yolov8n.pt'
model = YOLO(MODEL_PATH)
MODEL_PATH = '/tmp/best.pt'
modelPlate = YOLO('/tmp/best.pt')


# Modelo de solicitação para buscar a imagem do S3
class S3ImageRequest(BaseModel):
    bucket_name: str
    object_key: str

@app.post("/predict/")
async def predict_image(request: S3ImageRequest):
    try:
        response = s3_client.get_object(Bucket=request.bucket_name, Key=request.object_key)
        img_data = response['Body'].read()
        print ("Predict ", request.bucket_name)
        print ("Predict ", request.object_key)
        
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = model.predict(source=img)
        print ("Results", results)
        
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(annotated_img)
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG')
        buf.seek(0)
        
         # Define the new S3 object key for the annotated image
        annotated_key = 'annotated/' + os.path.basename(request.object_key)
        
        # Upload the annotated image back to S3
        s3_client.upload_fileobj(buf, request.bucket_name, annotated_key, ExtraArgs={'ContentType': 'image/jpeg'})
        
        # Return the URL of the uploaded annotated image
        file_url = f"https://{request.bucket_name}.s3.amazonaws.com/{annotated_key}"
        return {
            'statusCode': 200,
            'body': json.dumps({'file_url': file_url})
        }
        #return StreamingResponse(buf, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detectPlate/")
async def extract_plate(request: S3ImageRequest):
    try:
        print ("detectPlate !!")
        response = s3_client.get_object(Bucket=request.bucket_name, Key=request.object_key)
        img_data = response['Body'].read()
        print ("detectPlate ", request.bucket_name)
        print ("detectPlate ", request.object_key)
        
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = modelPlate(img)
        print ("Results", results)
        detections = results[0]

        for detection in detections:
            if detection.names[0] == 'placa':
                print ("inside if")
                x_min, y_min, x_max, y_max = map(int, detection.boxes[0].xyxy[0].tolist())
                #img = cv2.imread(image_path)
                plate_img = img[y_min:y_max, x_min:x_max]
                
                pil_image = Image.fromarray(plate_img)
                buf = io.BytesIO()
                pil_image.save(buf, format='JPEG')
                buf.seek(0)
            
                # Define the new S3 object key for the annotated image
                plate_key = 'plate/' + os.path.basename(request.object_key)
            
                # Upload the annotated image back to S3
                s3_client.upload_fileobj(buf, request.bucket_name, plate_key, ExtraArgs={'ContentType': 'image/jpeg'})
            
                # Return the URL of the uploaded annotated image
                file_url = f"https://{request.bucket_name}.s3.amazonaws.com/{plate_key}"
                return {
                'statusCode': 200,
                'body': json.dumps({'file_url': file_url})
                }
            else:
                print ("inside else")
                return {
                'statusCode': 402,
                'body': "PLATE not founc"
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exemplo de endpoint para verificar se a API está funcionando
@app.get("/")
async def read_root():
    return {"message": "API is working!!!!"}
