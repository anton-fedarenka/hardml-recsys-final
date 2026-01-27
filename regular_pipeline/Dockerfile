FROM python:3.12-bookworm

RUN mkdir /regular_pipeline
RUN mkdir /data

COPY regular_pipeline/requirements.txt .
COPY bandit.py .
COPY regular_pipeline/main.py /regular_pipeline

RUN pip install -r requirements.txt

# CMD [ "python", "main.py" ]