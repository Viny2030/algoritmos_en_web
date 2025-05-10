
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir seaborn

COPY requirements.txt .
COPY app.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
