FROM python:3.12-bookworm

WORKDIR /event_collector

COPY ../requirements.txt .
RUN pip install -r requirements.txt

COPY ../watched_filter.py ../models.py ./
COPY event_collector/main.py .

CMD [ "uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0" ]
