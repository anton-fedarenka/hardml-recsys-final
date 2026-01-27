FROM python:3.12-bookworm

WORKDIR /web_app

COPY webapp/requirements.txt .
RUN pip install -r requirements.txt

COPY webapp/app.py .
COPY webapp/static/ ./static/
COPY webapp/templates/ ./templates/

# CMD [ "python", "app.py"]
