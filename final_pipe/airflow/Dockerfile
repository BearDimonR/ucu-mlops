FROM apache/airflow:2.8.3-python3.10
ADD requirements.txt .
ENV PYTHONPATH="$PYTHONPATH:./"
RUN pip3 install -r requirements.txt