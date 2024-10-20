# Use a imagem base do AWS Lambda com Python 3.10
FROM public.ecr.aws/lambda/python:3.10

# Defina o diretório de trabalho onde o código e arquivos serão copiados
WORKDIR /var/task
RUN yum install -y libGL libGL-devel

# Instale as dependências necessárias
RUN pip install opencv-python-headless numpy torch torchvision Pillow boto3 ultralytics

# Copie o código da função Lambda para o diretório de trabalho
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Copie o modelo treinado para o diretório de trabalho
COPY yolov8_model.pt ${LAMBDA_TASK_ROOT}

# Comando padrão para executar a função Lambda
CMD ["lambda_function.lambda_handler"]
