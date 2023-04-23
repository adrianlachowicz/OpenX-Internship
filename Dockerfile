FROM python:3.7
EXPOSE 8080

WORKDIR /workspace
ADD . /workspace

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH="$PYTHONPATH:/workspace"
WORKDIR /workspace/src/gradio_app

CMD ["python3", "main.py"]
