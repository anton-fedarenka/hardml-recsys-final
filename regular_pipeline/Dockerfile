FROM python:3.12-bookworm

RUN mkdir /data
WORKDIR /regular_pipeline

COPY ../requirements.txt .
RUN pip install -r requirements.txt

COPY regular_pipeline/main.py .

CMD [ "python", "main.py" ]