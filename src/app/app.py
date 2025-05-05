import pickle

import torch
from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer

from src.app.utils import predict_text

app = Flask(__name__)

# Загрузка модели при запуске приложения
print("Загрузка модели...")
model_path = '../../models/rubert_hackothon'
tokenizer_path = '../../models/rubert_hackothon_tokenizer'
mlb_path = '../../models/label_binarizer.pkl'

# Загружаем токенайзер и модель
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    problem_type="multi_label_classification"
)

# Определяем устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")
model = model.to(device)
model.eval()  # Режим оценки (не обучения)

# Загружаем MultiLabelBinarizer
with open(mlb_path, 'rb') as f:
    mlb = pickle.load(f)

print(f"Модель загружена успешно! Количество классов: {len(mlb.classes_)}")


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        if text:
            # Делаем предсказание
            prediction = predict_text(text,
                                      tokenizer=tokenizer,
                                      model=model,
                                      device=device,
                                      mlb=mlb)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
