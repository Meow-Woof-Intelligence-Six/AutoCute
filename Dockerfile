# 1. 选择合适的基础镜像
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 将本地的 requirements.txt 文件复制到容器中
COPY requirements.txt /app/

COPY ./third_parties/ /app/third_parties/


# 4. 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# thundersvm
# cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
# cd ../python
# python setup.py install



# 5. 将当前目录下所有文件复制到容器的工作目录
COPY ./code/ /app/code/
COPY ./model/ /app/model/
COPY ./init.sh /app/init.sh
COPY ./train.sh /app/train.sh
COPY ./test.sh /app/test.sh
