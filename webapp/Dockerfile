FROM python:3.12-bookworm

WORKDIR /web_app

COPY webapp/requirements.txt .
RUN pip install -r requirements.txt

COPY webapp/static/ ./static/
COPY webapp/templates/ ./templates/

COPY webapp/app.py .

# CMD [ "python", "app.py"]
