# Use uma imagem base do AWS Lambda com Python 3.10
FROM public.ecr.aws/lambda/python:3.10
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    autoconf \
    automake \
    libtool \
    curl \
    wget \
    tar \
    libGL \
    libpng \
    libjpeg \
    libGL-devel \
    libpng-devel \
    libjpeg-devel \
    libtiff-devel \
    && yum clean all


# compile o Leptonica
RUN wget http://www.leptonica.org/source/leptonica-1.82.0.tar.gz && \
    tar -xvzf leptonica-1.82.0.tar.gz && \
    cd leptonica-1.82.0 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf leptonica-1.82.0 leptonica-1.82.0.tar.gz

# Configure o caminho
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/pkgconfig

# compile o Tesseract
RUN wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.3.0.tar.gz && \
    tar -xvzf 5.3.0.tar.gz && \
    cd tesseract-5.3.0 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf tesseract-5.3.0 tesseract-5.3.0.tar.gz

# Instale dependências Python
RUN pip install boto3 pytesseract Pillow opencv-python-headless ultralytics

RUN mkdir -p /usr/share/tessdata/

# Adicione os dados de treinamento do Tesseract

RUN curl -L https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata -o /usr/share/tessdata/eng.traineddata
#RUN curl -L https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata -o /usr/share/tessdata/eng.traineddata
#COPY eng.traineddata /usr/share/tessdata/
#COPY osd.traineddata /usr/share/tessdata/


# Verifique se o Tesseract e os dados foram instalados corretamente
RUN tesseract --version
RUN ls -la /usr/share/tessdata/
RUN chmod 775 /usr/share/tessdata/eng.traineddata
RUN chmod 775 /usr/share/tessdata


ENV TESSDATA_PREFIX=/usr/share/tessdata/
ENV PATH="/usr/local/bin:${PATH}"

# Defina o diretório de trabalho
WORKDIR /var/task

# Copie o código da função Lambda para o diretório de trabalho
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Comando padrão para executar a função Lambda
CMD ["lambda_function.lambda_handler"]

