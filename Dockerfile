FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Create the destination directory within the container
RUN mkdir -p /GAZ_MSA
COPY . /GAZ_MSA

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
