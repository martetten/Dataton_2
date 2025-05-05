# Gradient Seekers
Всем привет! Наша команда Gradient Seekers, и мы подготовили проект по теме «Тематическая классификация текстов»

## Задача «Тематическая классификация текстов»

## Цель и задачи
**Цель проекта** —  создание системы, которая определяет вероятность принадлежности текста к одной или нескольким тематикам из заданного списка.


**Задачи проекта**:   
1. Решить задачу классификации с пересекающимися классами. Разработать и обучить модель машинного обучения, способную анализировать текст и предсказывать вероятности его принадлежности к одной или нескольким тематикам.
2. Создать пользовательский интерфейс, который принимает текст на вход, отправляет его на классификацию и возвращает категорию, к которой относится текст.

## Описание данных 
Весь набор данных взят из соцсетей в исходном виде. Он состоит из 6 датасетов (каждый датасет - отдельный класс) 

Каждая строка файла - это некоторый пост из соцсети. Пост может содержать исходный текст (то, что написал сам пользователь), текст на картинке, текст речи из видео. Поэтому файл имеет 3 колонки, разделенные знаком ",".

Колонка `doc_text` содержит исходный текст самого поста.
Колонка `image2text` содержит текст, распознанный на изображении поста с помощью системы OCR.
Колонка `speech2text` содержит текст речи, распознанной с помощью системы ASR .

Некоторые посты могут не иметь те или иные поля, т.е. содержат пустой текст.

Классы (тематики текстов): спорт, юмор, реклама, соцсети, политика, личная жизнь.

## Описание модели

[`DeepPavlov/rubert-base-cased`](https://huggingface.co/DeepPavlov/rubert-base-cased) - это языковая модель на основе BERT, разработанная DeepPavlov и обученная на русской части Википедии и новостных данных. 

Основные плюсы модели:
- Высокое качество благодаря архитектуре BERT
- Поддержка контекстного понимания текста  (учитывает семантику слов в предложении, а не только их отдельные значения)
- Оптимизация для русского языка (она более эффективная для русского языка по сравнению с мультиязычными BERT-моделями)
- Поддержка токенизации с учётом регистра (cased)

## Реализация проекта
**Стек технологий**:  
- Python
- PyTorch
- Docker
- Flask
- HTML, CSS, JS
- Bootstrap


**Этапы работы:**
1. Предобработка данных (очистка, токенизация)
2. Оценка распределения данных по классам. Обучение нескольких моделей, их сравнение
3. Подбор гиперпараметров
4. Выбор лучшей модели и объяснение выбора
5. Предсказание на тестовых данных
6. Разработка UI интерфейса и Flask приложения
7. Тестирование

## Инструкция для запуска

### Запуск

Установить зависимости
```python
pip install -r requirements.txt
```

Поднять контейнер
```bash
docker compose up -d
```

Запустить Flask приложения  

```bash
cd src/app/
python app.py
```

## Результат
Приложение с использованием предобученной модели машинного обучения, которое дает возможность пользователю определить тематику введенного текста (от 2 до 30 слов): спорт, юмор, реклама, соцсети, политика, личная жизнь.

## Возможности развития проекта
1. Улучшение модели классификации
    - Увеличение датасета, расширение спектра категорий
    - Использование альтернативных моделей вместо DeepPavlov/rubert-base-cased для повышения точности (Например, DeepPavlov/rubert-large-cased, ai-forever/ruRoberta-large)
    - Дообучение модели на специфичных данных (создать размеченный датасет)


2. Улучшение UI/UX
    - Сохранение истории запросов и добавление возможности экспорта результатов (CSV или JSON)
    - Визуализация уверенности модели (графики, heatmap по темам)


## Команда

Мартынов Артем Васильевич [@martetten](https://github.com/martetten)  
**Team Lead, Data Scientist**  

Бек Владимир Эдуардович  [@VladimirBek](https://github.com/VladimirBek)  
**Data Scientist, App dev** 

Смолякова Евгения Владимировна [@EvgeniaWave](https://github.com/EvgeniaWave)  
**Data Scientist**  

Яровикова Анастасия Сергеевна  [@ynastt](https://github.com/ynastt)  
**Data Scientist, App dev**  

Серов Илья Алексеевич  [@EliSerov](https://github.com/EliSerov)  
**Data Scientist**   

Игнатьева Анастасия Юрьевна [@AnastasiaIgn](https://github.com/AnastasiaIgn)  
**Data Scientist**  





Project Organization
------------

```
Dataton2/
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py    
    │   ├── cleaning.py          
    │   ├── ingestion.py         
    │   ├── labeling.py          
    │   ├── splitting.py         
    │   └── validation.py        
    │
    ├── app                     # Flask app.
    │   ├── static  
    │   │   └── favicon-32x32.png
    │   ├── templates    
    │   │   └── index.html  
    │   └── app.py   
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── dataloader.py    
    │       ├── hyperparameters_tuning.py 
    │       ├── model.py         
    │       ├── predict.py       
    │       ├── preprocessing.py 
    │       └── train.py         
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
```


--------

