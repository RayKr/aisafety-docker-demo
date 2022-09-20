FROM --platform=linux/amd64 python:3.6.15-slim-buster
WORKDIR /home

COPY requirements.txt ./
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir -r requirements.txt

COPY ./ /home

CMD ["python", "main.py"]