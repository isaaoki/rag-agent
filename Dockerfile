FROM python:3.14

WORKDIR /app 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY src/ ./src/

EXPOSE 7860

CMD ["python", "src/main.py"]
