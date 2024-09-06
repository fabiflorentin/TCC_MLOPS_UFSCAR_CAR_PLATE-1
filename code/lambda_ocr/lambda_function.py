import boto3
import io
import cv2
import pytesseract
import os
import numpy as np
import subprocess

s3_client = boto3.client('s3')

YOLOV8_DIR = '/tmp'

# Set or modify the value of YOLO_CONFIG_DIR
os.environ['YOLO_CONFIG_DIR'] = "/tmp"
# Verify that the value has been updated
print(f"YOLO_CONFIG_DIR: {os.getenv('YOLO_CONFIG_DIR')}")

from ultralytics import YOLO

os.environ['TESSDATA_PREFIX'] = '/usr/share/tessdata/'
print(f"TESSDATA_PREFIX: {os.getenv('TESSDATA_PREFIX')}")

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    image_key = event['Records'][0]['s3']['object']['key']
    print ("OCR plate ", bucket_name)
    print ("OCR plate ", image_key)
   
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    plate_img = response['Body'].read()
    
    nparr = np.frombuffer(plate_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Aumenta o tamanho da imagem
    img_size = cv2.resize(img, None, fy=4, fx=4, interpolation=cv2.INTER_CUBIC)
    copia = img_size.copy()

    # Converter a imagem para escala de cinza
    gray_plate = cv2.cvtColor(img_size, cv2.COLOR_BGR2GRAY)

    # Operação morfológica de dilatação para preencher lacunas nas bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated_place = cv2.dilate(gray_plate, kernel, iterations=1)
    #show_image(dilated_place)

    cnts, _ = cv2.findContours(dilated_place, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_items_coordinates = []

    tessdata_path= os.path.join(os.getenv('TESSDATA_PREFIX'),"eng.traineddata")
    print ("Path", tessdata_path)
    if not os.path.isfile(tessdata_path):
        print ("Path not find", tessdata_path)
    

    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 40000:
            # Ajuste os valores de redução conforme necessário
            reduction_factor = 0.09

            new_w = int(w * (1.028 - reduction_factor))
            new_h = int(h * (0.83 - reduction_factor))

            # Calcular as novas coordenadas para manter a ROI centralizada
            new_x = x + (w - new_w) // 2
            new_y = y + (h - new_h) // 1

            # Certifique-se de que as novas coordenadas estejam dentro dos limites da imagem
            new_x = max(new_x, 0)
            new_y = max(new_y, 0)
            new_w = min(new_w, img_size.shape[1] - new_x)
            new_h = min(new_h, img_size.shape[0] - new_y)

            cv2.rectangle(copia, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 255), 3)
            roi = img_size[new_y:new_y+new_h, new_x:new_x+new_w]  # Recorte a região de interesse (ROI) da imagem original
            #show_image(roi)
            pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
            tessdata_dir_config = "--tessdata-dir /usr/share/tessdata/"
            tesseract_version = subprocess.getoutput("tesseract -v")
            print (f"Tess version: {tesseract_version}" )

            tes_langs = subprocess.getoutput("tesseract --tessdata-dir /usr/share/tessdata/ --list-langs")
            print (f"Tess langs: {tes_langs}" )

            tesseract_data_permission = subprocess.getoutput("ls -all /usr/share/tessdata/eng.traineddata")
            print (f"Tess permission: {tesseract_data_permission}" )
            
            tessdata_dir_config = "--tessdata-dir /usr/share/tessdata/ -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6"
            os.environ['TESSDATA_PREFIX'] = '/usr/share/tessdata/'

            print(f"TESSDATA_PREFIX: {os.getenv('TESSDATA_PREFIX')}")
            print (f"tessdata_dir_config: {tessdata_dir_config}")
            # Usar Tesseract OCR para extrair o texto da ROI
            #config = r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6"
            try:
                text = pytesseract.image_to_string(roi, lang="eng", config=tessdata_dir_config)
                print(f'Texto detectado: {text}')
                line_items_coordinates.append((new_x, new_y, new_w, new_h))
                print(f'x: {new_x}, y: {new_y}, w: {new_w}, h: {new_h}, text: {text}')
            except pytesseract.TesseractError as e:
                print(f"Erro 1 do Tesseract: {str(e)}")
            #text = pytesseract.image_to_string(roi, lang="eng", config=tessdata_dir_config)

            
