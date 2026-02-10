FROM python:3.12-bookworm

COPY regular_pipeline/requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /data
RUN mkdir /regular_pipeline

COPY bandit.py .
COPY regular_pipeline/main.py /regular_pipeline

# CMD [ "python", "main.py" ]