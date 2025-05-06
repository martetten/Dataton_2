#!/usr/bin/env python
# coding: utf-8

# # Партнерская задача: тематическая классификация текстов

# In[41]:


pip install -r requirements.txt


# In[42]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
from transformers import EarlyStoppingCallback
from pathlib import Path
from datetime import datetime
import pickle
import os


# In[43]:


import sys
print(sys.executable)


# ## 1. Исследование датасета и предобработка данных

# даны 6 датасетов (каждый датасет - отдельный класс) + описание
# 
# разделение "текст поста", video2text [OCR], speech2text [ASR]

# In[44]:


df1 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/1.csv')
df2 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/2.csv')
df3 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/3.csv')
df4 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/4.csv')
df5 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/5.csv')
df6 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/6.csv')


# In[45]:


df1['class'] = 1
df2['class'] = 2
df3['class'] = 3
df4['class'] = 4
df5['class'] = 5
df6['class'] = 6


# Объедним данные в общий general датафрейм

# In[46]:


df_gen = pd.concat([df1, df2, df3, df4, df5, df6]).reset_index()


# In[47]:


model_name = 'DeepPavlov/rubert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)


# In[48]:


combined_text = []

for i in range(len(df_gen)):
    # Объединяем значения из трех столбцов через пробел
    combined = str(df_gen['doc_text'].iloc[i]) + ' ' + \
                   str(df_gen['image2text'].iloc[i]) + ' ' + \
                   str(df_gen['speech2text'].iloc[i])
    combined_text.append(combined)


# In[49]:


df_gen['combined_text'] = combined_text


# In[50]:


texts = df_gen['combined_text']
labels = df_gen['class'].values - 1  # BERT ожидает классы 0-5 вместо 1-6


# In[51]:


train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels
)


# In[52]:


train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=64)


# In[53]:


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[54]:


train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)


# In[55]:


# Загрузка модели
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    ignore_mismatched_sizes=True
)


# In[56]:


output_dir = "../models/rubert_SL_datathon"  # Основная папка для модели
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}_tokenizer", exist_ok=True)


# In[57]:


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)

    # Дополнительная метрика: учитываем второй вариант
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1)
    top2_correct = 0
    for prob, label in zip(probs, p.label_ids):
        _, top2_idx = torch.topk(prob, k=2)
        if label in top2_idx:
            top2_correct += 1

    return {
        'accuracy': accuracy,
        'top2_accuracy': top2_correct / len(p.label_ids)
    }


# In[58]:


# Параметры обучения
training_args = TrainingArguments(
    output_dir=output_dir,  # Папка для чекпоинтов (будет создана)
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,  # Сохранять только лучшую модель
)

# Метрика для оценки
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Запуск обучения
trainer.train()


# In[59]:


model.save_pretrained(output_dir)
tokenizer.save_pretrained(f"{output_dir}_tokenizer")

print(f"Модель сохранена в: {output_dir}")
print(f"Токенизатор сохранен в: {output_dir}_tokenizer")


# Плюсы:
# - Accuracy растёт с 66.75% до 67.33%
# - Training Loss стабильно уменьшается (1.01 -> 0.51)
# 
# Минусы:
# - Validation Loss увеличился на 3-й эпохе (0.911 -> 0.980) - признаки переобучения
# - Accuracy на валидации выросла всего на 1% на последней эпохе

# In[60]:


def predict_with_two_results(model, tokenizer, texts, threshold=0.75):
    """Возвращает список, где каждый элемент:
    - [top_class] если уверенность >= threshold
    - [top_class, second_class] если уверенность < threshold
    """
    # Токенизация
    inputs = tokenizer(list(texts), truncation=True, padding=True, max_length=64, return_tensors="pt")

    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Обработка результатов
    results = []
    for prob in probs:
        top2_probs, top2_indices = torch.topk(prob, k=2)
        if top2_probs[0] >= threshold:
            results.append([top2_indices[0].item() + 1])  # +1 чтобы вернуть классы 1-6
        else:
            results.append([top2_indices[0].item() + 1, top2_indices[1].item() + 1])

    return results


# In[61]:


# Пример предсказания на тестовых данных
test_texts_list = test_texts.tolist()
predictions = predict_with_two_results(model, tokenizer, test_texts_list)


# In[62]:


# Вывод результатов для первых 10 примеров
for i in range(10):
    print(f"Текст: {test_texts_list[i][:50]}...")
    if len(predictions[i]) == 1:
        print(f"Предсказание: {predictions[i][0]} (уверенность >= 75%)")
    else:
        print(f"Предсказания: {predictions[i][0]} и {predictions[i][1]} (низкая уверенность)")
    print("-" * 50)


# Изменим гиперпараметры

# In[63]:


model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    hidden_dropout_prob=0.2,    # Увеличиваем dropout
    attention_probs_dropout_prob=0.2,
    ignore_mismatched_sizes=True
)


# In[64]:


training_args = TrainingArguments(
    output_dir=output_dir,  # Папка для чекпоинтов (будет создана)
    num_train_epochs=4,                    # Увеличим на 1 эпоху
    per_device_train_batch_size=16,        # Увеличим батч если позволяет память
    learning_rate=3e-5,                    # Увеличим с 2e-5 до 3e-5
    warmup_steps=100,                      # Уменьшим прогрев
    weight_decay=0.05,                     # Добавим регуляризацию
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=30,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True                              # Включим (есть GPU)
)


# Добавим в трейнер возможность новой остановки, в случае, если на новой эпохе показатели ухудшатся

# In[65]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# In[66]:


trainer.train()


# In[67]:


model.save_pretrained(output_dir)
tokenizer.save_pretrained(f"{output_dir}_tokenizer")

print(f"Модель сохранена в: {output_dir}")
print(f"Токенизатор сохранен в: {output_dir}_tokenizer")


# In[68]:


# Пример предсказания на тестовых данных
test_texts_list = test_texts.tolist()
predictions = predict_with_two_results(model, tokenizer, test_texts_list)


# In[69]:


# Вывод результатов для первых 10 примеров
for i in range(10):
    print(f"Текст: {test_texts_list[i][:50]}...")
    if len(predictions[i]) == 1:
        print(f"Предсказание: {predictions[i][0]} (уверенность >= 75%)")
    else:
        print(f"Предсказания: {predictions[i][0]} и {predictions[i][1]} (низкая уверенность)")
    print("-" * 50)


# Изменим значение остановки, чтобы она происходила уже после первого ухудшения

# In[70]:


run_dir = model_dir / "training_runs" / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
run_dir.mkdir(parents=True, exist_ok=True)


# In[71]:


training_args = TrainingArguments(
    output_dir=output_dir,  # Папка для чекпоинтов (будет создана)
    report_to="tensorboard",               # Автоматический логи
    logging_strategy="steps",              # Логировать по шагам
    logging_steps=50,                      # Частота логирования
    num_train_epochs=10,                   # Увеличим до 10 эпох
    per_device_train_batch_size=16,        # Увеличим батч если позволяет память
    learning_rate=5e-6,                    # Уменьшаем learning rate
    warmup_steps=100,                      # Уменьшим прогрев
    weight_decay=0.05,                     # Добавим регуляризацию
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,                # Accuracy чем выше тем лучше
    fp16=True                              # Включим (есть GPU)
)


# In[72]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1,
            early_stopping_threshold=0.01,  # Минимальное улучшение
        )
    ]
)


# In[73]:


trainer.train()


# Результаты показывают, что модель начала переобучаться уже на второй эпохе (Validation Loss увеличился с 1.089 до 1.208 при (небольшой) стагнации Accuracy), расчет остановлен

# In[74]:


predictions = predict_with_two_results(model, tokenizer, test_texts_list)


# In[249]:


# Вывод результатов для первых 5 примеров
for i in range(4, 9):
    print(f"Текст: {test_texts_list[i][:50]}...")
    if len(predictions[i]) == 1:
        print(f"Предсказание: {predictions[i][0]} (уверенность >= 75%)")
    else:
        print(f"Предсказания: {predictions[i][0]} и {predictions[i][1]} (низкая уверенность)")
    print("-" * 50)

