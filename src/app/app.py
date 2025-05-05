from flask import Flask, render_template, request
import pickle
import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Загрузка модели и векторизатора при старте
# with open('models/model.pkl', 'rb') as f:
#     model_data = pickle.load(f)
#     model = model_data['model']
#     vectorizer = model_data['vectorizer']
#     categories = model_data['categories']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        # Векторизация текста
        # X = vectorizer.transform([text])
        # Предсказание
        # pred = model.predict(X)[0]
        # prediction = categories[pred]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)