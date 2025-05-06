#!/usr/bin/env python
# coding: utf-8

# # Партнерская задача: тематическая классификация текстов

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
import inflect
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from scipy.sparse import hstack
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.sparse import hstack
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
from transformers import EarlyStoppingCallback
from pathlib import Path
from datetime import datetime

import sys
print(sys.executable)


# ## 1. Исследование датасета и предобработка данных

# даны 6 датасетов (каждый датасет - отдельный класс) + описание
# 
# разделение "текст поста", video2text [OCR], speech2text [ASR]



df1 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/1.csv')
df2 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/2.csv')
df3 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/3.csv')
df4 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/4.csv')
df5 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/5.csv')
df6 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/6.csv')




print(df1.info())
print(df2.info())
print(df3.info())
print(df4.info())
print(df5.info())
print(df6.info())



df1['class'] = 1
df2['class'] = 2
df3['class'] = 3
df4['class'] = 4
df5['class'] = 5
df6['class'] = 6


# Объедним данные в общий general датафрейм


df_gen = pd.concat([df1, df2, df3, df4, df5, df6]).reset_index()


df_gen.tail()


# Смотрим количество пропусков


df_gen.info()


# Функция для преобразования эмоджи в их текстовые описания (перечисляет их через запятую)


def extract_emojis(text):
    if not isinstance(text, str):
        return np.nan

    # Находим все эмодзи в тексте
    emoji_chars = [c for c in text if c in emoji.EMOJI_DATA]

    if not emoji_chars:
        return np.nan

    # Преобразуем эмодзи в текстовые описания (без : и _)
    emoji_descriptions = [
        emoji.demojize(e, delimiters=(" ", " "))
        .replace(":", "").replace("_", " ")
        .strip()
        for e in emoji_chars
    ]

    # Объединяем через запятую
    return ", ".join(emoji_descriptions)


# Функция для удаления эмоджи из текта

def remove_emojis(text):
    if not isinstance(text, str):
        return np.nan
    return emoji.replace_emoji(text, replace="")


# Функция для предварительной очистки и нормализации текста

def base_clean_text(text: str):
    if not isinstance(text, str):
        return np.nan

    text = text.lower()

    # Удаляем URL, хэштеги, упоминания ДО split()
    text = re.sub(r'(https?://\S+)|(#\w+)|(@\w+)', '', text)

    # Удаляем скобки/кавычки, мешающие обработке
    text = re.sub(r'[()\[\]{}"\']', '', text)

    # Оставляем только буквы и пробелы
    text = re.sub(r'[^\sа-яёa-z]', '', text)

    # Убираем повторы букв (например, "привееет" -> "привет")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    stop_words = set(stopwords.words('russian'))
    temp = inflect.engine()
    words = []

    for word in text.split():
        if word.isdigit():
            words.append(temp.number_to_words(word))
        elif word and word not in stop_words:  # игнорируем пустые строки и стоп-слова
            words.append(word)

    return ' '.join(words) if words else np.nan


# Функция для приведения слов к их корневой форме

def stem_russian_text(text):
    stemmer = RussianStemmer()
    words = text.split()
    stemmed_words = []
    for word in words:
        if re.match('[а-яА-Я]', word):
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        else:
            stemmed_words.append(word)
    return ' '.join(stemmed_words)


# Применяем функции на агреггированном датасете

df_gen['emojis_doc_text'] = df_gen['doc_text'].apply(extract_emojis)


df_gen['emojis_doc_text'] = df_gen['emojis_doc_text'].fillna('').values


df_gen['doc_text'] = df_gen['doc_text'].apply(remove_emojis)


for col in ['doc_text', 'image2text', 'speech2text']:
    df_gen[f'cleaned_{col}'] = [base_clean_text(x) if isinstance(x, str) else np.nan
                                 for x in df_gen[col]]


# Смотрим пример преобразованного текста

df_gen.iloc[0]['cleaned_doc_text']


for col in ['doc_text', 'image2text', 'speech2text']:
    df_gen[f'stemmed_{col}'] = [stem_russian_text(x) if isinstance(x, str) else np.nan
                                 for x in df_gen[f'cleaned_{col}']]


def clean_text_for_spacy(text: str):
    if not isinstance(text, str):
        return np.nan

    # Удаляем URL, хэштеги, упоминания (но сохраняем пунктуацию!)
    text = re.sub(r'(https?://\S+)|(#\w+)|(@\w+)', '', text)

    # Удаляем только проблемные символы (например, математические)
    text = re.sub(r'[{}<>$%^&*|\\]', '', text)

    # Приводим к нижнему регистру (spaCy сам обрабатывает регистр)
    text = text.lower()

    # Убираем повторы букв (но сохраняем пунктуацию)
    text = re.sub(r'([а-яёa-z])\1{2,}', r'\1', text)

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else np.nan


for col in ['doc_text', 'image2text', 'speech2text']:
    df_gen[f'SpaCy_{col}'] = [clean_text_for_spacy(x) if isinstance(x, str) else np.nan
                                 for x in df_gen[col]]


df_gen.head()




def analyze_text_column(column):
    all_words = ' '.join(column.fillna('')).split()
    word_freq = Counter(all_words)
    print(f"Top 20 слов для {column.name}:")
    print(word_freq.most_common(20))




analyze_text_column(df_gen['stemmed_doc_text'])
analyze_text_column(df_gen['stemmed_image2text'])
analyze_text_column(df_gen['stemmed_speech2text'])


# Длина текстов
df_gen['doc_length'] = df_gen['stemmed_doc_text'].str.split().str.len()
df_gen['image_length'] = df_gen['stemmed_image2text'].str.split().str.len()
"""
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.boxplot(df_gen['doc_length'].dropna())
plt.title('Длина doc_text')

plt.subplot(122)
plt.boxplot(df_gen['image_length'].dropna())
plt.title('Длина image2text')
plt.show()
"""

# ## 2. Feature Engineering


for col in df_gen[['doc_text', 'image2text', 'speech2text', 'emojis_doc_text', 'cleaned_doc_text', 'cleaned_image2text', 'cleaned_speech2text', 'stemmed_doc_text', 'stemmed_image2text', 'stemmed_speech2text', 'SpaCy_doc_text', 'SpaCy_image2text', 'SpaCy_speech2text']]:
    df_gen[col] = df_gen[col].fillna('').values



tfidf = TfidfVectorizer(
    max_features=5000,       # Ограничиваем количество фич
    min_df=5,                # Игнорировать слова, встречающиеся <5 раз
    max_df=0.7,              # Игнорировать слова, встречающиеся >70% документов
    ngram_range=(1, 2)       # Учитывать словосочетания (1-2 слова)
)

tfidf_low_quality = TfidfVectorizer(
    max_features=1000,
    min_df=10,  # Более строгий фильтр для редких слов
    max_df=0.9,
    token_pattern=r'\b[а-яё]{4,}\b'  # Только слова из 4+ русских букв
)


tfidf_doc = tfidf.fit_transform(df_gen['stemmed_doc_text'].fillna(''))
tfidf_image = tfidf_low_quality.fit_transform(df_gen['stemmed_image2text'].fillna(''))
tfidf_speech = tfidf_low_quality.fit_transform(df_gen['stemmed_speech2text'].fillna(''))


# Кластеризация для каждого типа текста
n_clusters = 6

kmeans_doc = KMeans(n_clusters=n_clusters).fit(tfidf_doc)
kmeans_image = KMeans(n_clusters=n_clusters).fit(tfidf_image)
kmeans_speech = KMeans(n_clusters=n_clusters).fit(tfidf_speech)


# Добавляем результаты в датафрейм
df_gen['cluster_doc'] = kmeans_doc.labels_
df_gen['cluster_image'] = kmeans_image.labels_
df_gen['cluster_speech'] = kmeans_speech.labels_


print("Сходство кластеров doc_text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_doc']))

print("Сходство кластеров image2text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_image']))

print("Сходство кластеров speech2text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_speech']))


def custom_tokenizer(text):
    # Удаляем спецсимволы (но сохраняем слова с дефисами и апострофами)
    text = re.sub(r'[^\w\s-]', '', text.lower())

    # Токенизация с учетом русской морфологии
    tokens = word_tokenize(text, language='russian')

    # Фильтрация стоп-слов и коротких токенов
    russian_stopwords = set(stopwords.words('russian'))
    tokens = [token for token in tokens 
              if token not in russian_stopwords 
              and len(token) > 2 
              and not token.isdigit()]

    return tokens


tfidf_advanced = TfidfVectorizer(
    tokenizer=custom_tokenizer,          # Наш токенизатор
    analyzer='word',                     # Анализ по словам
    ngram_range=(1, 2),                  # Учитываем словосочетания
    min_df=3,                            # Игнорировать редкие слова (<3 документов)
    max_df=0.85,                         # Игнорировать слишком частые слова (>85% документов)
    sublinear_tf=True,                   # Логарифмическое масштабирование TF
    smooth_idf=True,                     # Сглаживание IDF
    norm='l2',                           # Нормализация векторов
    lowercase=True                       # Приводить к нижнему регистру (уже делаем в токенизаторе)
)


tfidf_doc_adv = tfidf_advanced.fit_transform(df_gen['SpaCy_doc_text'].fillna(''))


kmeans_doc_adv = KMeans(n_clusters=n_clusters).fit(tfidf_doc_adv)


df_gen['cluster_doc_adv'] = kmeans_doc_adv.labels_


print("Сходство кластеров doc_text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_doc_adv']))


# Для основного текста
tsne = TSNE(n_components=2, random_state=42)
tfidf_doc_2d = tsne.fit_transform(tfidf_doc.toarray())
"""
plt.figure(figsize=(12, 6))
plt.scatter(tfidf_doc_2d[:, 0], tfidf_doc_2d[:, 1], 
            c=df_gen['class'], cmap='tab10', alpha=0.6)
plt.title('t-SNE: Основной текст (цвета = классы)')
plt.colorbar(ticks=range(1,7))
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
"""
sources = {
    'Основной текст': tfidf_doc,
    'OCR текст': tfidf_image, 
    'ASR текст': tfidf_speech,
    'SpaCy текст': tfidf_doc_adv
}
"""
for (name, data), ax in zip(sources.items(), axes.flatten()):
    emb_2d = TSNE(n_components=2).fit_transform(data.toarray())
    ax.scatter(emb_2d[:,0], emb_2d[:,1], c=df_gen['class'], cmap='tab10', alpha=0.6)
    ax.set_title(f't-SNE: {name}')
"""

# Визуализация данных в 2D также показывает отсутствие видимых кластеров

results = []
for name, data in sources.items():
    # Кластеризация
    kmeans = KMeans(n_clusters=6, random_state=42).fit(data)

    # Оценка
    sil_score = silhouette_score(data, kmeans.labels_)
    rand_score = adjusted_rand_score(df_gen['class'], kmeans.labels_)

    results.append({
        'Источник': name,
        'Silhouette': sil_score,
        'Adjusted Rand': rand_score
    })

pd.DataFrame(results).sort_values('Adjusted Rand', ascending=False)


# Результаты теста на силуэт и скорректированная Accuracy показывают низкие результаты
# 
# При этом для OCR и ASR результаты на силуэт достаточно высоки (Accuracy ниже 1%), что может говорить о том, что кол-во кластеров ~угадано, но все они перепутаны

# Выведем по 10 топ слов для SpaCy_doc_text

# Для tfidf_advanced (SpaCy)
feature_names = tfidf_advanced.get_feature_names_out()
tfidf_matrix = tfidf_doc_adv.tocsr()  # Конвертируем в CSR формат

for class_idx in range(6):
    # Получаем индексы документов класса
    doc_indices = df_gen[df_gen['class'] == class_idx+1].index
    # Вычисляем среднее TF-IDF по классу
    class_tfidf = tfidf_matrix[doc_indices].mean(axis=0).A1

    top_words_idx = class_tfidf.argsort()[-10:][::-1]
    print(f"\nКласс {class_idx+1} топ-10 слов:")
    print([feature_names[i] for i in top_words_idx])


# ## 3.  Подбор признаков, их анализ и оценка важности

# Попробуем улучшить результаты кластеризации, изменив характеристики токенизаторов


tfidf = TfidfVectorizer(
    max_features=5000,       # Ограничиваем количество фич
    min_df=10,                # Игнорировать слова, встречающиеся <5 раз
    max_df=0.6,              # Игнорировать слова, встречающиеся >70% документов
    ngram_range=(1, 3)       # Учитывать словосочетания (1-2 слова)
)

tfidf_low_quality = TfidfVectorizer(
    max_features=500,
    min_df=10,  # Более строгий фильтр для редких слов
    max_df=0.8,
    token_pattern=r'\b[а-яё]{4,}\b'  # Только слова из 4+ русских букв
)

tfidf_advanced = TfidfVectorizer(
    tokenizer=custom_tokenizer,          # Наш токенизатор
    analyzer='word',                     # Анализ по словам
    ngram_range=(1, 3),                  # Учитываем словосочетания
    min_df=6,                            # Игнорировать редкие слова (<3 документов)
    max_df=0.85,                         # Игнорировать слишком частые слова (>85% документов)
    sublinear_tf=True,                   # Логарифмическое масштабирование TF
    smooth_idf=True,                     # Сглаживание IDF
    norm='l2',                           # Нормализация векторов
    lowercase=True                       # Приводить к нижнему регистру (уже делаем в токенизаторе)
)




tfidf_doc = tfidf.fit_transform(df_gen['stemmed_doc_text'].fillna(''))
tfidf_image = tfidf_low_quality.fit_transform(df_gen['stemmed_image2text'].fillna(''))
tfidf_speech = tfidf_low_quality.fit_transform(df_gen['stemmed_speech2text'].fillna(''))
tfidf_doc_adv = tfidf_advanced.fit_transform(df_gen['SpaCy_doc_text'].fillna(''))





print("Форма TF-IDF матрицы:", tfidf_doc.shape)




# Токенизация для SpaCy
nlp = spacy.load("ru_core_news_sm")
def spacy_tokenize(text):
    if isinstance(text, float) and np.isnan(text):  # Пропускаем NaN
        return []

    doc = nlp(text)
    return [
        token.text for token in doc
        if not token.is_space  # Игнорируем пробелы
    ]



df_gen['SpaCy_doct_text_tokens'] = df_gen['SpaCy_doc_text'].apply(spacy_tokenize)




print("Пример токенов:", df_gen['SpaCy_doct_text_tokens'].iloc[0][:10]) # Первые 10


# ## 4. Обучение нескольких моделей, их сравнение


X_combined = hstack([
    tfidf_doc * 0.3,
    tfidf_doc_adv * 0.6, # Делаем вес текста для SpaCy выше
    tfidf_image * 0.1,  # Понижаем вес OCR текста
    tfidf_speech * 0.05  # И ASR тоже
])




X_train, X_test, y_train, y_test = train_test_split(
    X_combined, 
    df_gen['class'], 
    test_size=0.2,
    stratify=df_gen['class']
)



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "SVM": SVC(class_weight='balanced')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}:\n", classification_report(y_test, y_pred))


# Статистики для классов 2 и 4 хуже всего
# 
# Модели LogReg и SVM лучше, но не достаточно хорошо, всего 65% accuracy
# 
# Построим матрицу, чтобы визуализировать, какие классы больше всего путает модель SVM


"""
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(models["SVM"], X_test, y_test)
plt.title("Confusion Matrix for SVM")
plt.show()
"""

# Пробуем построить те же модели, предварительно создав метапризнаки

# Создаем метапризнаки (и сразу их скейлируем для конкретной работы моделей)

meta_features = pd.DataFrame({
    'doc_length': df_gen['doc_text'].str.len(),          # Длина исходного текста
    'num_words': df_gen['stemmed_doc_text'].str.split().str.len(),  # Количество слов
    'num_emojis': df_gen['emojis_doc_text'].str.count(' ') + 1,    # Число эмодзи
    'has_image': df_gen['image2text'].notna().astype(int),          # Наличие картинки
    'has_speech': df_gen['speech2text'].notna().astype(int)         # Наличие аудио
})


# Проверяем отсутствие пропусков


print(meta_features.isna().sum())


scaler = StandardScaler()
meta_scaled = scaler.fit_transform(meta_features)


# Объединяем с текстовыми признаками
X_combined_meta = hstack([X_combined, meta_scaled])



X_train, X_test, y_train, y_test = train_test_split(
    X_combined_meta, 
    df_gen['class'], 
    test_size=0.2,
    stratify=df_gen['class']
)



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "SVM": SVC(class_weight='balanced')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}:\n", classification_report(y_test, y_pred))


# После добавления мета признаков качество моделей либо почти не изменилось (для Log Reg accuracy стала 0.62), либо сильно ухудшилось (для SVM стало 0.55)
# 
# Вернемся на этап с токенизацией и попробуем BERT



model_name = 'DeepPavlov/rubert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)


combined_text = []

for i in range(len(df_gen)):
    # Объединяем значения из трех столбцов через пробел
    combined = str(df_gen['doc_text'].iloc[i]) + ' ' + \
                   str(df_gen['image2text'].iloc[i]) + ' ' + \
                   str(df_gen['speech2text'].iloc[i])
    combined_text.append(combined)



df_gen['combined_text'] = combined_text



texts = df_gen['combined_text']
labels = df_gen['class'].values - 1  # BERT ожидает классы 0-5 вместо 1-6


train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels
)


train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=64)


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



train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)



# Загрузка модели
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    ignore_mismatched_sizes=True
)
print("Последний слой инициализирован заново - это нормально для transfer learning")


project_root = Path.cwd().parent
model_dir = project_root / "models" / "bert_classifier"
run_dir = model_dir / "training_runs" / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
run_dir.mkdir(parents=True, exist_ok=True)



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


# Параметры обучения
training_args = TrainingArguments(
    output_dir=str(run_dir / "checkpoints"),  # Путь для чекпоинтов
    logging_dir=str(run_dir / "logs"),        # Путь для логов
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
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


# Плюсы:
# - Accuracy растёт с 65.42% до 68.58%
# - Training Loss стабильно уменьшается (1.01 -> 0.59)
# 
# Минусы:
# - Validation Loss увеличился на 3-й эпохе (0.947 -> 0.979) - признаки переобучения
# - Accuracy на валидации выросла всего на 2% на последней эпохе

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



# Пример предсказания на тестовых данных
test_texts_list = test_texts.tolist()
predictions = predict_with_two_results(model, tokenizer, test_texts_list)


# Вывод результатов для первых 5 примеров
for i in range(5):
    print(f"Текст: {test_texts_list[i][:50]}...")
    if len(predictions[i]) == 1:
        print(f"Предсказание: {predictions[i][0]} (уверенность >= 75%)")
    else:
        print(f"Предсказания: {predictions[i][0]} и {predictions[i][1]} (низкая уверенность)")
    print("-" * 50)


# Изменим гиперпараметры


run_dir = model_dir / "training_runs" / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
run_dir.mkdir(parents=True, exist_ok=True)



model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    hidden_dropout_prob=0.2,    # Увеличиваем dropout
    attention_probs_dropout_prob=0.2,
    ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    output_dir=str(run_dir / "checkpoints"),  # Путь для чекпоинтов
    logging_dir=str(run_dir / "logs"),        # Путь для логов
    num_train_epochs=4,                    # Увеличим на 1 эпоху
    per_device_train_batch_size=16,        # Увеличим батч если позволяет память
    learning_rate=3e-5,                    # Увеличим с 2e-5 до 3e-5
    warmup_steps=100,                      # Уменьшим прогрев
    weight_decay=0.05,                     # Добавим регуляризацию
    logging_steps=30,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True                              # Включим (есть GPU)
)


# Добавим в трейнер возможность новой остановки, в случае, если на новой эпохе показатели ухудшатся

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


trainer.train()


# Пример предсказания на тестовых данных
test_texts_list = test_texts.tolist()
predictions = predict_with_two_results(model, tokenizer, test_texts_list)


# Вывод результатов для первых 5 примеров
for i in range(4, 9):
    print(f"Текст: {test_texts_list[i][:50]}...")
    if len(predictions[i]) == 1:
        print(f"Предсказание: {predictions[i][0]} (уверенность >= 75%)")
    else:
        print(f"Предсказания: {predictions[i][0]} и {predictions[i][1]} (низкая уверенность)")
    print("-" * 50)


# Изменим значение остановки, чтобы она происходила уже после первого ухудшения

run_dir = model_dir / "training_runs" / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
run_dir.mkdir(parents=True, exist_ok=True)


training_args = TrainingArguments(
    output_dir=str(run_dir / "checkpoints"),  # Путь для чекпоинтов
    logging_dir=str(run_dir / "logs"),        # Путь для логов
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


trainer.train()


# Результаты показывают, что модель начала переобучаться уже на второй эпохе (Validation Loss увеличился с 1.089 до 1.208 при (небольшой) стагнации Accuracy), расчет остановлен

predictions = predict_with_two_results(model, tokenizer, test_texts_list)


# Вывод результатов для первых 5 примеров
for i in range(4, 9):
    print(f"Текст: {test_texts_list[i][:50]}...")
    if len(predictions[i]) == 1:
        print(f"Предсказание: {predictions[i][0]} (уверенность >= 75%)")
    else:
        print(f"Предсказания: {predictions[i][0]} и {predictions[i][1]} (низкая уверенность)")
    print("-" * 50)



# Попробовать кластеризацию, насколько реально выполнена разметка/разделение датасетов

# 3 версии датасета:
# - только посты
# - посты + дополнения
# - все источники с маркерами [post] text, [OCR] text, [ASR] text

# взвешенная конкатенация (тект + OCR + ASR)
def combine_sources(row):
    main_text = row['user_text']
    supplements = ' '.join([row['ocr_text'], row['asr_text']])
    return f"{main_text} [SUPP] {supplements}"  # Маркер для дополнений


# ## 6. Выбор лучшей модели и объяснение выбора

# ## 7. Предсказание на тестовых данных
