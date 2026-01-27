FROM python:3.12-bookworm

RUN mkdir /recommendations
RUN mkdir /data

COPY recommendations/requirements.txt .
RUN pip install -r requirements.txt

COPY ./models.py .
COPY ./bandit.py .
COPY recommendations/main.py /recommendations

# CMD [ "uvicorn", "main:app", "--port", "5001", "--host", "0.0.0.0" ]
