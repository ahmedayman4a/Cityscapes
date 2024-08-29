FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]

# Example usage: 
# docker run --rm cityscapes:0.1 --model_path data/unet_dice_v0.7_augment_dice_best.pth --dataset_path /home/ahmed/Downloads/kaggle/input/cityscapes/Cityspaces
