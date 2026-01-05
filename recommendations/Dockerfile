FROM python:3.12-bookworm

WORKDIR /recommendations

COPY ../requirements.txt .
RUN pip install -r requirements.txt

COPY ../models.py ../watched_filter.py ./
COPY recommendations/main.py .

CMD [ "uvicorn", "main:app", "--port", "5001", "--host", "0.0.0.0" ]
