FROM python:3.11-slim


COPY ./requirements.txt /tmp/

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

EXPOSE 5000

# Запуск приложения
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app.app:app"]
