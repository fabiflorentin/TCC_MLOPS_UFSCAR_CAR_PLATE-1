# Use a imagem base do Ubuntu 20.04
FROM ubuntu:20.04

# Defina o argumento para frontend não interativo
ARG DEBIAN_FRONTEND=noninteractive

# Instale Python e outras dependências necessárias
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    && apt-get clean

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos do projeto para o diretório de trabalho
COPY . /app

# Copie o modelo treinado para o diretório /tmp
COPY yolov8n.pt /tmp/yolov8n.pt
COPY best.pt /tmp/best.pt

# Instale as dependências Python
RUN pip3 install -r requirements.txt

# Exponha a porta que a aplicação usará
EXPOSE 8000

# Defina o comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
