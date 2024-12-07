# Используйте официальный Python-образ
FROM python:3.9-slim

# Установите рабочую директорию
WORKDIR /app

# Скопируйте зависимости
COPY requirements.txt .

# Установите зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скопируйте весь код проекта в контейнер
COPY . .

# Укажите команду по умолчанию (например, для запуска скрипта обработки видео)
CMD ["python", "src/process_video.py", "data/input_video.mp4", "data/output_video/processed_video.mp4"]