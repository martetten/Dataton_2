FROM python:3.11-slim

WORKDIR /app

# Копирование requirements.txt и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY app.py .
COPY templates/ ./templates/

# Копирование файлов модели
COPY rubert_hackothon/ ./rubert_hackothon/
COPY label_binarizer.pkl .

# Открытие порта для Flask
EXPOSE 5000

# Запуск приложения
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]