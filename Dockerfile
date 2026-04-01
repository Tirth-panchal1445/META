FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install gymnasium numpy flask
EXPOSE 7860
CMD ["python", "inference.py"]
