FROM python:3.12-bookworm

RUN mkdir /event_collector

COPY event_collector/requirements.txt .
RUN pip install -r requirements.txt

COPY ./models.py ./
COPY event_collector/main.py /event_collector

# CMD [ "uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0" ]
