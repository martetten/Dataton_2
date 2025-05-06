#!/usr/bin/env python
# coding: utf-8

# # Партнерская задача: тематическая классификация текстов

# In[2]:


# pip install -r requirements.txt


# In[3]:


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
from sklearn.feature_extraction.text import CountVectorizer
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


# In[4]:


import sys
print(sys.executable)


# ## 1. Исследование датасета и предобработка данных

# даны 6 датасетов (каждый датасет - отдельный класс) + описание
# 
# разделение "текст поста", video2text [OCR], speech2text [ASR]

# In[5]:


df1 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/1.csv')
df2 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/2.csv')
df3 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/3.csv')
df4 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/4.csv')
df5 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/5.csv')
df6 = pd.read_csv('https://github.com/martetten/Dataton_2/raw/main/data/raw/6.csv')


# In[6]:


print(df1.info())
print(df2.info())
print(df3.info())
print(df4.info())
print(df5.info())
print(df6.info())


# In[7]:


# df1 = df1.drop(columns='class')


# In[8]:


df1['class'] = 1
df2['class'] = 2
df3['class'] = 3
df4['class'] = 4
df5['class'] = 5
df6['class'] = 6


# Объедним данные в общий general датафрейм

# In[9]:


df_gen = pd.concat([df1, df2, df3, df4, df5, df6]).reset_index()


# In[10]:


df_gen.tail()


# Смотрим количество пропусков

# In[11]:


df_gen.info()


# Функция для преобразования эмоджи в их текстовые описания (перечисляет их через запятую)

# In[12]:


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

# In[13]:


def remove_emojis(text):
    if not isinstance(text, str):
        return np.nan
    return emoji.replace_emoji(text, replace="")


# Функция для предварительной очистки и нормализации текста

# In[14]:


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

# In[15]:


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

# In[16]:


df_gen['emojis_doc_text'] = df_gen['doc_text'].apply(extract_emojis)


# In[17]:


df_gen['emojis_doc_text'] = df_gen['emojis_doc_text'].fillna('').values


# In[18]:


df_gen['doc_text'] = df_gen['doc_text'].apply(remove_emojis)


# In[19]:


for col in ['doc_text', 'image2text', 'speech2text']:
    df_gen[f'cleaned_{col}'] = [base_clean_text(x) if isinstance(x, str) else np.nan
                                 for x in df_gen[col]]


# Смотрим пример преобразованного текста

# In[20]:


df_gen.iloc[0]['cleaned_doc_text']


# In[21]:


# Топ 20 слов
all_words = ' '.join(df_gen['cleaned_doc_text'].dropna().astype(str)).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=[count for word, count in top_words], 
            y=[word for word, count in top_words], 
            palette='viridis')
plt.title('Топ 20 самых частых слов')
plt.xlabel('Частота')
plt.ylabel('Слово')
plt.show()


# In[22]:


vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=20)
X = vectorizer.fit_transform(df_gen['cleaned_doc_text'].dropna().astype(str))
bigrams = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
top_bigrams = sorted(bigrams, key=lambda x: x[1], reverse=True)[:20]

plt.figure(figsize=(10, 6))
sns.barplot(x=[count for bigram, count in top_bigrams], 
            y=[bigram for bigram, count in top_bigrams], 
            palette='magma')
plt.title('Топ 20 самых частых биграм')
plt.xlabel('Частота')
plt.ylabel('Биграм')
plt.show()


# In[23]:


for col in ['doc_text', 'image2text', 'speech2text']:
    df_gen[f'stemmed_{col}'] = [stem_russian_text(x) if isinstance(x, str) else np.nan
                                 for x in df_gen[f'cleaned_{col}']]


# In[24]:


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


# In[25]:


for col in ['doc_text', 'image2text', 'speech2text']:
    df_gen[f'SpaCy_{col}'] = [clean_text_for_spacy(x) if isinstance(x, str) else np.nan
                                 for x in df_gen[col]]


# In[26]:


df_gen.head()


# In[27]:


def analyze_text_column(column):
    all_words = ' '.join(column.fillna('')).split()
    word_freq = Counter(all_words)
    print(f"Top 20 слов для {column.name}:")
    print(word_freq.most_common(20))


# In[28]:


analyze_text_column(df_gen['stemmed_doc_text'])
analyze_text_column(df_gen['stemmed_image2text'])
analyze_text_column(df_gen['stemmed_speech2text'])


# In[29]:


# Длина текстов
df_gen['doc_length'] = df_gen['stemmed_doc_text'].str.split().str.len()
df_gen['image_length'] = df_gen['stemmed_image2text'].str.split().str.len()

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.boxplot(df_gen['doc_length'].dropna())
plt.title('Длина doc_text')

plt.subplot(122)
plt.boxplot(df_gen['image_length'].dropna())
plt.title('Длина image2text')
plt.show()


# In[30]:


plt.figure(figsize=(12, 4))
sns.histplot(df_gen['doc_length'], bins=50, kde=True, color='skyblue')
plt.title('Распределение длин текстов по кол-ву слов')
plt.xlabel('Длина doc_text')
plt.ylabel('Количество')
plt.show()


# In[31]:


plt.figure(figsize=(12, 4))
sns.histplot(df_gen[df_gen['image_length'] > 0]['image_length'], bins=50, kde=True, color='salmon')
plt.title('Распределение длин текстов по кол-ву слов')
plt.xlabel('Длина image2text')
plt.ylabel('Количество')
plt.show()


# Визуально распределение не нормальное

# In[32]:


plt.figure(figsize=(12, 6))
sns.histplot(data=df_gen, x='doc_length', hue='class', bins=70, kde=True, palette='viridis')
plt.title('Распределение количества слов по классам')
plt.xlabel('Количество слов')
plt.ylabel('Частота')
plt.xlim(0, 250) # ограничим диапазон по значимым частотам
plt.show()


# Видим, что чаще всего в посте 10 слов

# ## 2. Feature Engineering

# In[33]:


for col in df_gen[['doc_text', 'image2text', 'speech2text', 'emojis_doc_text', 'cleaned_doc_text', 'cleaned_image2text', 'cleaned_speech2text', 'stemmed_doc_text', 'stemmed_image2text', 'stemmed_speech2text', 'SpaCy_doc_text', 'SpaCy_image2text', 'SpaCy_speech2text']]:
    df_gen[col] = df_gen[col].fillna('').values


# In[34]:


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


# In[35]:


tfidf_doc = tfidf.fit_transform(df_gen['stemmed_doc_text'].fillna(''))
tfidf_image = tfidf_low_quality.fit_transform(df_gen['stemmed_image2text'].fillna(''))
tfidf_speech = tfidf_low_quality.fit_transform(df_gen['stemmed_speech2text'].fillna(''))


# In[36]:


# Кластеризация для каждого типа текста
n_clusters = 6

kmeans_doc = KMeans(n_clusters=n_clusters).fit(tfidf_doc)
kmeans_image = KMeans(n_clusters=n_clusters).fit(tfidf_image)
kmeans_speech = KMeans(n_clusters=n_clusters).fit(tfidf_speech)


# In[37]:


# Добавляем результаты в датафрейм
df_gen['cluster_doc'] = kmeans_doc.labels_
df_gen['cluster_image'] = kmeans_image.labels_
df_gen['cluster_speech'] = kmeans_speech.labels_


# In[38]:


print("Сходство кластеров doc_text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_doc']))

print("Сходство кластеров image2text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_image']))

print("Сходство кластеров speech2text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_speech']))


# In[39]:


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


# In[40]:


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


# In[41]:


tfidf_doc_adv = tfidf_advanced.fit_transform(df_gen['SpaCy_doc_text'].fillna(''))


# In[42]:


kmeans_doc_adv = KMeans(n_clusters=n_clusters).fit(tfidf_doc_adv)


# In[43]:


df_gen['cluster_doc_adv'] = kmeans_doc_adv.labels_


# In[44]:


print("Сходство кластеров doc_text с исходными классами:",
      adjusted_rand_score(df_gen['class'], df_gen['cluster_doc_adv']))


# In[45]:


# Для основного текста
tsne = TSNE(n_components=2, random_state=42)
tfidf_doc_2d = tsne.fit_transform(tfidf_doc.toarray())

plt.figure(figsize=(12, 6))
plt.scatter(tfidf_doc_2d[:, 0], tfidf_doc_2d[:, 1], 
            c=df_gen['class'], cmap='tab10', alpha=0.6)
plt.title('t-SNE: Основной текст (цвета = классы)')
plt.colorbar(ticks=range(1,7))
plt.show()


# In[46]:


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sources = {
    'Основной текст': tfidf_doc,
    'OCR текст': tfidf_image, 
    'ASR текст': tfidf_speech,
    'SpaCy текст': tfidf_doc_adv
}

for (name, data), ax in zip(sources.items(), axes.flatten()):
    emb_2d = TSNE(n_components=2).fit_transform(data.toarray())
    ax.scatter(emb_2d[:,0], emb_2d[:,1], c=df_gen['class'], cmap='tab10', alpha=0.6)
    ax.set_title(f't-SNE: {name}')


# Визуализация данных в 2D также показывает отсутствие видимых кластеров

# In[47]:


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

# In[48]:


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

