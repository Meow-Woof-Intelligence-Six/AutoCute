#!/bin/bash
# 由于镜像太大，需要init的时候再安装依赖
pip install --no-cache-dir autogluon==1.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple