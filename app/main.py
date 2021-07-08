from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle5 as pickle
import pandas as pd
import nltk
nltk.download('stopwords')
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
stemmer=PorterStemmer()
from utils import get_base_url, allowed_file, and_syntax



# setup the webservver
# port = 12345
# base_url = get_base_url(port)
app = Flask(__name__)


IMAGE_FOLDER=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER

with open('fake_news_predictor.pkl','rb') as f1:
	model=pickle.load(f1)

with open('tfidf1.pkl','rb') as f2:
	tf1=pickle.load(f2)
    


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/result",methods=["GET","POST"])
@cross_origin()
def result():
	if request.method=="POST":
		word_list=[]
		news=(request.form["News"])
		words=re.sub('[^a-zA-Z]',' ',news)
		words=words.lower().split()
		words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
		words=' '.join(words)
		word_list.append(words)
		tf_vec=TfidfVectorizer(max_features=5000,ngram_range=(1,3),vocabulary = tf1.vocabulary_)
		Embedded_News=tf_vec.fit_transform(word_list).toarray()
		prediction=model.predict(Embedded_News)

		output=""
		if prediction==0:
			output="Fake"
			img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'Sad_Emoji.png')
		else:
			output="Real"
			img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'Smiling_Emoji.png')


		return render_template('home.html',prediction_text=f'This is a {output} News',image=img_filename)

	return render_template("home.html")



# if __name__ == "__main__":
    # change the code.ai-camp.org to the site where you are editing this file.
#    print("Try to open\n\n    https://cocalc8.ai-camp.org" + base_url + '\n\n')
    # remove debug=True when deploying it
#    app.run(host = '0.0.0.0', port=port, debug=True)
#    import sys; sys.exit(0)
