from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model
with open('svm.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html', analysis_result=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    
    # Preprocess text
    transformed_text = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(transformed_text)[0]
    
    return render_template('index.html', analysis_result=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)