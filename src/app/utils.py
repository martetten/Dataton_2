import numpy as np
import torch

CLASS_MAPPING = {
    1: 'Соцсети',
    2: 'Личная жизнь',
    3: 'Политика',
    4: 'Реклама',
    5: 'Спорт',
    6: 'Юмор'
}


def predict_text(text, threshold=0.25, max_len=512,
                 tokenizer=None,
                 model=None,
                 device=None,
                 mlb=None):
    """
    Функция для предсказания классов текста

    Args:
        text (str): Текст для классификации
        threshold (float): Порог вероятности для присвоения класса
        max_len (int): Максимальная длина последовательности

    Returns:
        list: Список названий предсказанных классов
    """
    # Токенизация текста
    encoding = tokenizer(
        [text],
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).to(device)

    # Получение предсказаний
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        predictions = (probs >= threshold).cpu().numpy()

    # Преобразование в индексы классов
    pred_indices = np.where(predictions[0])[0]

    # Преобразуем numpy int64 в обычные питоновские int и затем в имена классов
    if len(pred_indices) > 0:
        # Преобразуем numpy.int64 в стандартные питоновские int
        pred_indices = [int(idx) for idx in pred_indices]

        # Добавим отладочную информацию
        print(f"Предсказанные индексы: {pred_indices}")
        print(f"Доступные классы: {mlb.classes_}")
        print(f"Количество классов: {len(mlb.classes_)}")

        # Проверка индексов и получение имен классов
        try:
            pred_classes = [int(mlb.classes_[idx]) for idx in pred_indices]
            mapped_classes = [CLASS_MAPPING[idx] for idx in pred_classes]
            if not mapped_classes:
                result = "Класс не определён"
            else:
                result = ', '.join(mapped_classes)

            return result
        except IndexError as e:
            print(f"Ошибка при доступе к классам: {e}")
            # В случае ошибки возвращаем индексы (для отладки)
            return [f"Класс #{idx}" for idx in pred_indices]

    return []
