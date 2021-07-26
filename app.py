from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# from sklearn.externals import joblib
import joblib
import pickle

# load the model from disk
#filename = 'nlp_model.pkl'
#clf = pickle.load(open(filename, 'rb'))
#cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)
clf = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
		if request.method == 'POST':
			message = request.form['tenure in days']
			data = [message]
			#data = np.array(data)
			int_features = [int(x) for x in request.form.values()]
			final_features = [np.array(int_features)]						
			my_prediction = clf.predict(final_features)
		return render_template('result.html',prediction = my_prediction)
if __name__ == '__main__':
	app.run(debug=True)
