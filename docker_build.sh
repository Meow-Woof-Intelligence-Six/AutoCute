sudo docker buildx build  --platform linux/amd64  --build-arg IMAGE_NAME=nvidia/cuda  -t bdc2025 .
sudo docker save -o 喵汪特工队.tar bdc2025:latest
sudo docker load -i 喵汪特工队.tar