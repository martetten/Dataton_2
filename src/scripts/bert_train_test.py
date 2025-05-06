import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import re
import emoji
import inflect
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import f1_score

from torch.utils.data import Dataset


df1 = pd.read_csv(os.path.join('..', '..', 'data', 'raw', '1.csv'))
df2 = pd.read_csv(os.path.join('..', '..', 'data', 'raw', '2.csv'))
df3 = pd.read_csv(os.path.join('..', '..', 'data', 'raw', '3.csv'))
df4 = pd.read_csv(os.path.join('..', '..', 'data', 'raw', '4.csv'))
df5 = pd.read_csv(os.path.join('..', '..', 'data', 'raw', '5.csv'))
df6 = pd.read_csv(os.path.join('..', '..', 'data', 'raw', '6.csv'))

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
df_gen = pd.concat([df1, df2, df3, df4, df5, df6]).reset_index(drop=True)
df_gen

df = df_gen.replace('NaN', np.nan)

# Объединяем первые три колонки
# Функция для объединения строк с учетом NaN значений
def combine_columns(row):
    values = [row['doc_text'], row['image2text'], row['speech2text']]
    # Фильтруем None и NaN значения
    values = [str(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return ' | '.join(values) if values else np.nan

# Создаем новую колонку с объединенным текстом
df['text'] = df.apply(combine_columns, axis=1)

# Создаем новый DataFrame только с объединенной колонкой и классом
result_df = df[['text', 'class']]

# Выводим результат
print(result_df)


nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = base_clean_text(text)

    return text


def base_clean_text(text: str):
    text = text.lower()
    stop_words = set(stopwords.words('russian'))
    temp = inflect.engine()
    words = []
    for word in text.split():
        word = re.sub('http\S+', '', word)
        word = re.sub('[^\sа-яёЁА-Яa-zA-Z]', '', word)
        if word.isdigit():
            words.append(temp.number_to_words(word))
        else:
            if word not in stop_words:
                words.append(word)

    text = ' '.join(words)
    return text
def emojis_words(text):
    # Модуль emoji: преобразование эмоджи в их словесные описания
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Редактирование текста путём замены ":" и" _", а так же - путём добавления пробела между отдельными словами
    text = text.replace(":", "").replace("_", " ")
    return text
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
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.FloatTensor(self.labels[idx])  # Используем FloatTensor для мультиметочной классификации
        return item

    def __len__(self):
        return len(self.labels)


class BertClassifier:

    def __init__(self, model_path, tokenizer_path, n_classes=44, epochs=7, 
                 model_save_path='model_save', tokenizer_save_path='tokenizer_path', 
                 force_cpu=False, batch_size=16, max_len=512):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model_save_path = model_save_path
        self.tokenizer_save_path = tokenizer_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.n_classes = n_classes
        self.batch_size = batch_size

        # Определяем устройство
        if force_cpu:
            self.device = torch.device("cpu")
            print("Принудительно используется CPU")
        else:
            # Попробуем инициализировать на GPU, но будем готовы переключиться на CPU
            try:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    # Попробуем загрузить модель на GPU
                    self.model = BertForSequenceClassification.from_pretrained(
                        model_path, 
                        num_labels=n_classes,
                        problem_type="multi_label_classification"
                    ).to(self.device)
                    # Проверим, есть ли достаточно памяти, выделив тестовый тензор
                    test_tensor = torch.zeros((batch_size, max_len), device=self.device)
                    del test_tensor  # Освобождаем память
                    print("Используется GPU")
                else:
                    self.device = torch.device("cpu")
                    print("GPU недоступен, используется CPU")
                    self.model = BertForSequenceClassification.from_pretrained(
                        model_path, 
                        num_labels=n_classes,
                        problem_type="multi_label_classification"
                    ).to(self.device)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # При ошибке памяти переключаемся на CPU
                print("Недостаточно памяти GPU, переключение на CPU")
                torch.cuda.empty_cache()  # Очистка кэша GPU
                self.device = torch.device("cpu")
                # Заново инициализируем модель на CPU
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=n_classes,
                    problem_type="multi_label_classification"
                ).to(self.device)

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=self.n_classes)
        self.model.to(self.device)


    def tokenize_texts(self, texts, tokenizer, max_length):
        tokenized_texts = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return tokenized_texts

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # Не добавляем токены, как рекомендовано в исправлениях
        train_encodings = self.tokenize_texts(X_train, self.tokenizer, self.max_len)
        valid_encodings = self.tokenize_texts(X_valid, self.tokenizer, self.max_len)

        # create datasets
        self.train_set = CustomDataset(train_encodings, y_train)
        self.valid_set = CustomDataset(valid_encodings, y_valid)

        # create data loaders с меньшим размером батча на CPU
        if self.device.type == 'cpu':
            # Меньший размер батча для CPU
            cpu_batch_size = max(1, self.batch_size // 2)
            print(f"Используется уменьшенный размер батча на CPU: {cpu_batch_size}")
            self.train_loader = DataLoader(self.train_set, batch_size=cpu_batch_size)
            self.valid_loader = DataLoader(self.valid_set, batch_size=cpu_batch_size)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size)
            self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size)

        # helpers initialization с меньшим learning rate на CPU
        if self.device.type == 'cpu':
            self.optimizer = AdamW(self.model.parameters(), lr=1e-5)  # Меньший lr для CPU
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

    def fit(self):
        self.model.train()
        losses = 0

        for data in self.train_loader:
            self.optimizer.zero_grad()
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["labels"].to(self.device)

            # Для мультиметочной классификации labels должны быть float и иметь форму [batch_size, n_classes]
            # Если ваши метки не в таком формате, их нужно преобразовать

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            losses += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss = losses / len(self.train_loader)
        return train_loss

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        threshold = 0.2

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float().cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        # Объединяем предсказания и метки
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Вычисляем разные типы F1-метрик
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        print(f"F1-micro: {f1_micro:.4f}, F1-macro: {f1_macro:.4f}")
        return f1_micro, f1_macro

    def train(self):

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss = self.fit()
            print(f'Train loss {train_loss}')
            if hasattr(self, 'valid_loader'):
                f1_micro, f1_macro = self.evaluate()
                print(f'Validation F1 micro: {f1_micro}, F1 macro: {f1_macro}')

        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.tokenizer_save_path)


    def predict(self, texts: list, threshold=0.5):
        """
        Метод для предсказания меток в задаче мультиметочной классификации.

        Args:
            texts (list): Список текстов для классификации
            threshold (float): Порог вероятности для присвоения метки (0.0-1.0)

        Returns:
            list: Список списков индексов классов для каждого текста
                Пустой список означает, что текст не относится ни к одному классу
        """
        self.model.eval()  # Переключаем в режим оценки

        # Обработка текстов батчами
        batch_size = 16
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.max_len,
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                ).to(self.device)

                outputs = self.model(**encoding)
                logits = outputs.logits

                # Применяем sigmoid для получения вероятностей для каждого класса
                probabilities = torch.sigmoid(logits)

                # Применяем пороговое значение
                batch_preds = (probabilities >= threshold).cpu().numpy()
                all_predictions.extend(batch_preds)

        # Преобразуем в список индексов для каждого текста
        result = []
        for pred in all_predictions:
            # Получаем индексы классов, где предсказание превышает порог
            class_indices = np.where(pred)[0].tolist()
            result.append(class_indices)  # Может быть пустым списком, если нет классов

        return result


df = result_df
# Проверка данных
print(f"Размер датасета: {df.shape}")
print(f"Количество уникальных классов: {df['class'].nunique()}")
print(f"Распределение классов:\n{df['class'].value_counts()}")


grouped_texts = df.groupby('text')['class'].apply(list).reset_index()
print(f"Количество уникальных текстов после группировки: {len(grouped_texts)}")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    grouped_texts['text'], 
    grouped_texts['class'],
    test_size=0.2,
    random_state=42
)

# Преобразование списков классов в бинарные метки
mlb = MultiLabelBinarizer()
y_train_binary = mlb.fit_transform(y_train)
y_test_binary = mlb.transform(y_test)

print(f"Количество классов после преобразования: {len(mlb.classes_)}")
print(f"Форма бинарных меток для обучения: {y_train_binary.shape}")
with open(os.path.join('..', '..', 'models', 'label_binarizer.pkl'), 'wb') as f:
    pickle.dump(mlb, f)

classifier = BertClassifier(
    model_path="DeepPavlov/rubert-base-cased",
    tokenizer_path="DeepPavlov/rubert-base-cased",
    n_classes=len(mlb.classes_),
    epochs=3,
    model_save_path=os.path.join('..', '..', 'models', 'rubert_hackothon'),
    tokenizer_save_path=os.path.join('..', '..', 'models', 'rubert_hackothon_tokenizer')
)

# Подготовка данных
classifier.preparation(
    X_train.tolist(),
    y_train_binary.tolist(), 
    X_test.tolist(),
    y_test_binary.tolist()  
)

# Обучение модели
classifier.train()

# Пример использования для предсказания
texts_to_predict = ["Решили прогуляться с семьей по торговому центру. Обожаю такие семейные посиделки и все такое. Когда проголодались, вспомнили про Вкусс-Вилла. В нем всегда все свежее и полезное"]
predictions = classifier.predict(texts_to_predict)

# Получение оригинальных меток классов
for text, pred_indices in zip(texts_to_predict, predictions):
    if pred_indices:  # Если есть хотя бы один класс
        pred_classes = [mlb.classes_[idx] for idx in pred_indices]
        print(f"Текст: {text}")
        print(f"Предсказанные классы: {pred_classes}")
    else:
        print(f"Текст: {text}")
        print("Текст не относится ни к одному из классов")
