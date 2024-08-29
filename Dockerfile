FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY data_handling /app/data_handling
COPY model /app/model
COPY test /app/test
COPY main.py /app/main.py

ENTRYPOINT ["python", "main.py"]

# Example usage: 

# docker run --rm \
#     -v /home/ahmed/brightskies/Cityscapes/data:/container/model \
#     -v /home/ahmed/Downloads/kaggle/input/cityscapes/Cityspaces:/container/dataset \
#     ahmedayman4a/cityscapes \
#     --model_path /container/model/unet_dice_v0.7_augment_dice_best.pth \
#     --dataset_path /container/dataset
