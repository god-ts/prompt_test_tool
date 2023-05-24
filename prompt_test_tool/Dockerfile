# 使用官方Python基础镜像
FROM python:3.8-slim-buster

# 设置工作目录
WORKDIR /app

# 将requirements.txt复制到工作目录
COPY requirements.txt .

# 安装依赖项
RUN pip config set global.index-url https://pypi.douban.com/simple/ --trusted-host pypi.douban.com
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U scikit-learn

# 将当前目录的所有文件复制到工作目录
COPY . .

# 暴露端口
EXPOSE 8786

# 运行Flask应用
CMD ["gunicorn", "-b", "0.0.0.0:8786", "--timeout", "180", "app:app"]

