# 基础镜像
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 不再copy代码，避免每次代码修改都得构建巨大的镜像。
# 以后只有修改端口与requirements内容时才需要重新构建镜像
# 本镜像将相对稳定的作为模型运行环境。
# 把当前目录所有文件copy到/workspace下
#COPY . .

# 删除config.py文件
#RUN rm config.py

# 安装依赖
RUN pip install -r requirements.txt

# 容器内部暴露18080端口
EXPOSE 18080

# 设置环境变量
ENV FLASK_APP=web_service.py

# 运行flask web服务
CMD ["flask", "run", "--host=0.0.0.0", "--port=18080"]
