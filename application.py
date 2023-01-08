from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/quality_index")
def quality_index():
    return render_template("quality_index.html")

@app.route("/nba_stats")
def nba_stats():
    return render_template("nba_stats.html")

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")

@app.route("/consumption")
def consumption():
    return render_template("consumption.html")

@app.route("/minesweeper")
def minesweeper():
    return render_template("minesweeper.html")

@app.route("/topic_modelling", methods=["POST","GET"])
def get_corpora():
    if request.method == "GET":
        return render_template("topic_modelling.html")
    else:
        corpora = []
        for i in range(3):
            if not eval("request.form.get('text"+str(i)+"')"):
                return render_template("failure.html")
            corpora.append(word_tokenize(eval("request.form.get('text"+str(i)+"')")))
        #clean words 
        doc_no = len(corpora)
        for doc in range(doc_no):
            corpora[doc] = [WordNetLemmatizer().lemmatize(word.lower()) for word in corpora[doc] if word.isalpha()]
        for doc in range(doc_no):
            corpora[doc] = [word for word in corpora[doc] if word not in stopwords.words("english")]
        dictionary = Dictionary(corpora)
        corpus = [dictionary.doc2bow(corpus) for corpus in corpora]
        tfidf = TfidfModel(corpus)
        corpus_list = []
        for i in corpus:
            corpus_list.append(sorted(tfidf[i],key=lambda x:x[1],reverse = True))
        topics = []
        for i in corpus_list:
            try:
                topics.append(dictionary.get(i[0][0]))
            except:
                topics.append("NA")
        return render_template("topic_modelling_result.html",topics = topics)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)