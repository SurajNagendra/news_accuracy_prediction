import requests
import re
import feedparser
import csv
from bs4 import BeautifulSoup
from eventregistry import *
import pandas as pd
from prediction import *
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions, KeywordsOptions, MetadataOptions
from rake_nltk import Rake

from flask import Flask, render_template, request
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


# INIT ALL ML
print("loading tensorflow  model")
sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = loadML()


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/scrape", methods=["POST"])
def scrape():
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2018-11-16',
        iam_apikey='4kxtefSt-VgDt3LGbteO7tv0eAczVWdvXJcMIKhHdJfo',
        url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
    )

    if(request.method == "POST"):

        # response = natural_language_understanding.analyze(
        #     url=request.form["keyword"],
        #     features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=4))).get_result()
        # phrases = [request.form["keyword"]]
        # keywords = []
        # for i in range(len(phrases)):
        #     keywords = keywords + phrases[i].split()
        # phrases = phrases[0].split()
        response = natural_language_understanding.analyze(
            url=request.form["keyword"],
            features=Features(metadata=MetadataOptions())).get_result()

        article_title = response["metadata"]["title"]

        response2 = natural_language_understanding.analyze(
            text=article_title,
            features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=15))).get_result()

        keywords = []
        for keyword in response2['keywords']:
            if keyword['relevance'] > 0.5 and len(keywords) < 8:
                keywords.append(keyword['text'])

        new_keywords = []
        for i in range(len(keywords)):
            new_keywords = new_keywords + keywords[i].split()

        # r = Rake()

        # r.extract_keywords_from_text(article_title)
        # phrases = r.get_ranked_phrases()
        # print(phrases)
        # new_phrases = []
        # for i in range(len(phrases)):
        #     new_phrases = new_phrases + phrases[i].split()
        # new_phrases = phrases[:15]
        # for i in range(len(phrases)):
        #     if(len(phrases[i]) >= 3):
        #         new_phrases = new_phrases + phrases[i].split()
        #     else:
        #         new_phrases = new_phrases + phrases[i]
        # article_title = response["metadata"]["title"]
        # print(article_title)
        # new_phrases = article_title.split()
        # for i in range(len(response["keywords"])):
        #     if(response["keywords"][i]["relevance"] >= 0.4):
        #         new_phrases = new_phrases + [response["keywords"][i]["text"]]
        # keywords = []
        # for keyword in response['keywords']:
        #     keywords.append(keyword['text'])
        # new_phrases = keywords

    print(new_keywords)

    print('type of phrases:')
    print(type(new_keywords))
    api_key = 'e7c28375-a0b6-4566-b1c4-36b2af9f0009'
    er = EventRegistry(apiKey=api_key)

    q = QueryArticlesIter(
        keywords=new_keywords,
        keywordsLoc="title",
        sourceUri=QueryItems.OR(['indianexpress.com',
                                 'thehindu.com', 'ndtv.com', ' indiatoday.intoday.in', 'news18.com',
                                 'timesofindia.indiatimes.com', 'firstpost.com', 'deccanchronicle.com',
                                 'infowars.com', 'huffingtonpost.in']))

    csv_columns = ['uri', 'lang', 'isDuplicate', 'date', 'time', 'dateTime', 'dataType', 'sim',
                   'url', 'title', 'body', 'source', 'authors', 'image', 'eventUri', 'sentiment', 'wgt']
    csv_file = "test.csv"
    news_articles = []
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for article in q.execQuery(er, sortBy="rel", maxItems=500):
                # print(article)
                news_articles.append(article)
                writer.writerow(article)
    except IOError:
        print("I/O error")

    Body_ID = []
    f = pd.read_csv("test.csv")

    # taking column title from test.csv and creating test_stances_unlabeled.csv
    keep_col = ['title']
    new_f = f[keep_col]
    new_f = new_f.rename(columns={'title': 'Headline'})
    for i in range(new_f['Headline'].count()):
        Body_ID.append(i+1)
    new_f['Body ID'] = Body_ID
    new_f.to_csv("test_stances_unlabeled.csv", index=False)

    # taking column body from test.csv and creating test_bodies.csv
    keep_col = ['body']
    idx = 0
    new_f = f[keep_col]
    new_f = new_f.rename(columns={'body': 'articleBody'})
    new_f.insert(loc=idx, column='Body ID', value=Body_ID)
    new_f.to_csv("test_bodies.csv", index=False)

    runModel(sess, keep_prob_pl, predict, features_pl,
             bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    return render_template('downloading.html', news_articles=news_articles)


@app.route("/stances", methods=["POST"])
def stance():
    colnames = ['stance']
    data = pd.read_csv('predictions_test.csv', names=colnames)
    stances = data.stance.tolist()

    return render_template('stances.html', stances=stances)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = pd.read_csv('predictions_test.csv')
    stance = file['Stance'].value_counts()
    dictionary = {}
    for key in stance.keys():
        dictionary.update({key: stance[key]})
    stance = []
    total_count = 0
    for value in dictionary.values():
        total_count = total_count + value
    for key in dictionary.keys():
        stance.append({'stance': key, 'count': (
            dictionary[key]/total_count)*100})

    print(stance)
    return render_template('bargraph.html', stance=stance)


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.run(debug=True)
