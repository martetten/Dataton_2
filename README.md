# Gradient Seekers
Всем привет! Наша команда Gradient Seekers, и мы подготовили проект по теме «Тематическая классификация текстов»

## Тема «Тематическая классификация текстов»


## Актуальность / Проблема


## Цель и задачи
**Цель проекта** —  

**Задачи проекта**:   
1.   
2. 

## Описание данных и модели 
даны 6 датасетов (каждый датасет - отдельный класс) + описание
разделение "текст поста", video2text [OCR], speech2text [ACR]
Тематики: спорт, юмор, реклама, соцсети, политика, личная жизнь

## Реализация проекта
**Стек технологий**:  
- Python
- PyTorch
- Docker
- PostgreSQL
- Rabbit MQ
- aiogram


**Этапы работы:**
1. Предобработка данных
2. Feature Engineering
3. Подбор признаков, их анализ и оценка важности
4. Обучение нескольких моделей, их сравнение
5. Подбор гиперпараметров
6. Выбор лучшей модели и объяснение выбора
7. Предсказание на тестовых данных
8. Разработка базы данных
9. Разработка телеграм-бота
10. Тестирование

## Инструкция для запуска

#### Настройка переменных окружения
В корневой директории проекта создать ```.env``` файл, в котором будут храниться переменные окружения. Файл необходимо заполнить переменными, которые указаны в файле ```.env.example```.  

Пример записи в ```.env```:  
```text
BOT_TOKEN = <API token>
```

### Запуск

Для работы notebook
```python
pip install -r requirements.txt
```

Поднять все контейнеры
```bash
docker compose up -d
```
Зайти в один из контейнеров приложения и запустить миграции
```bash
docker exec -ti service_bot bash
alembic upgrade head
```
Заполнить данными таблицу с экспонатами, для корректной работы приложения и выйти из контейнера
```
python app/fill_table.py
```

## Результат
Чат-бот с использованием предобученной модели машинного обучения, который анализируюет тональность текста (от 2 до 30 слов) и определяет тематку: спорт, юмор, реклама, соцсети, политика, личная жизнь
[@ссылка]

## Возможности развития проекта
1. Текст:
    - 
    - 

2. Текст
    - 
    - 
    - 
    - 


## Команда

Бек Владимир Эдуардович  [@VladimirBek](https://github.com/VladimirBek)  
**Data Scientist, App dev** 

Смолякова Евгения Владимировна [@EvgeniaWave](https://github.com/EvgeniaWave)  
**Data Scientist**  

Яровикова Анастасия Сергеевна  [@ynastt](https://github.com/ynastt)  
**Data Scientist, Telegram bot developer**  

Серов Илья Алексеевич  [@EliSerov](https://github.com/EliSerov)  
**Data Scientist**   

Мартынов Артем Васильевич [@martetten](https://github.com/martetten)  
**Team Lead, Data Scientist**  

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
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
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
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>
