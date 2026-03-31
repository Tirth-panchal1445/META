FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install gymnasium numpy flask
EXPOSE 5000
CMD ["python", "inference.py"]
