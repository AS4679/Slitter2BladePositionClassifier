FROM python:3.9.7

SHELL ["/bin/bash", "--login", "-c"]
RUN apt-get update -y
RUN apt-get install -y nano

#Python env setup
RUN mkdir -p /home/service
WORKDIR /home/service
COPY requirements.txt /home/service
# COPY ../DigiOCVIP /home/service
RUN git config --global http.sslVerify false
RUN pip install --upgrade pip --no-cache-dir -r requirements.txt

COPY . /home/service