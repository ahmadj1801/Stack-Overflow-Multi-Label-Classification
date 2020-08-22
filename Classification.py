"""
                Ahmad Jawaad Shah
              COMP316 NLP/ML Project
       Tagging of Stack Overflow Questions. """

import csv
import re
import matplotlib.pyplot as plot
import nltk
import sklearn
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

questionList = []  # List of every Question
tagList = []  # Large List of every tag
uniqueTags = []  # Individual tags
uniqueQuestions = []  # Non Duplicate Questions
uniqueQuestionsTags = []  # Tags for the Non Duplicate Questions
tagCounts = {}  # (Dict) Tags and their respective frequencies/counts
num_tags_per_question = {}  # (Dict) How many tags per question


# Using Regular Expressions to remove square brackets
# and split into individual tags
def clean_and_split_tag(tags):
    tags = re.sub('(\[|\])', '', tags)
    tags = tags.strip()
    tags = tags.split(",")
    num_tags = tags.__len__()
    if num_tags in num_tags_per_question:
        num_tags_per_question[num_tags] = num_tags_per_question.get(num_tags) + 1
    else:
        num_tags_per_question[num_tags] = 1
    for i in range(tags.__len__()):
        tags[i] = tags[i].strip()
    return tags


# Reading data from the csv
def extract_data(read):
    duplicates = 0
    c = 1
    for row in read:
        title = row['title']
        tags = row['tags']
        tl = clean_and_split_tag(tags)
        if title.lower() in questionList:  # Count all duplicate Questions
            duplicates = duplicates + 1
        else:
            uniqueQuestions.append(title.lower())
            uniqueQuestionsTags.append(tl)
        questionList.append(title.lower())  # All the Questions from the training set
        tagList.append(tl)  # All the tags associated to the question at that index
        for tag in tl:
            if tag not in uniqueTags:
                uniqueTags.append(tag)
                tagCounts[tag] = 1
            else:
                tagCounts[tag] = tagCounts.get(tag) + 1
        c = c + 1
    print("Total Number of Duplicate Questions = ", duplicates)
    print("Total Number of Unique Tags = ", uniqueTags.__len__(), '\n')


# Create and Return a Pandas data frame
def create_data_frame():
    # Create a reader to read csv
    reader = csv.DictReader(open("Data/train.csv", 'rt', encoding='utf-8'))
    # Extract data into data structures
    extract_data(reader)
    # Clean the Data
    pre_process_data()
    # Create a data frame
    cols = ['Question', 'Tags']
    data = []
    pos = 0
    for q in uniqueQuestions:
        sub_data = [q]
        s = ''
        for tag in uniqueQuestionsTags[pos]:
            s = s + ' ' + tag
        sub_data.append(s.strip())
        data.append(sub_data)
        pos = pos + 1
    # Data frame with data
    data_frame = pd.DataFrame(data, columns=cols)
    # Return our Data Frame
    return data_frame


# Data Pre processing
# 1. Make all questions lower case for unification
# 2. Remove Punctuation using regular expressions
# 3. Remove stop words using NLTK's stop words
# 4. Stem each question using a Porter Stemmer
def pre_process_data():
    # Using NLTK's list of  English stop words
    stop_words = set(stopwords.words('english'))
    # print(stop_words)
    i = 0
    while i < uniqueQuestions.__len__():
        # Obtain Question
        text = "" + uniqueQuestions[i]
        # 1. Convert to lower case
        text = text.lower()
        # 2. Remove punctuation i.e Parenthesis, Commas and Question Marks
        text = re.sub('(\?|\,|\(|\))', " ", text)
        # Change what's to what is
        text = re.sub(r"what's", "what is", text)
        # Change can't to can not
        text = re.sub(r"can't", "can not", text)
        # replace all n't containing words
        text = re.sub(r"n't", " not", text)
        # 3. Remove Stop Words
        wt = word_tokenize(text)
        new_str = ''
        for w in wt:
            if w not in stop_words:
                new_str = new_str + ' ' + w
        text = new_str
        # 4. Stem Question
        uniqueQuestions[i] = stem_sentence(text)
        i = i + 1


# Using Porter Stemming on each Question
def stem_sentence(t):
    tokenisedWords = word_tokenize(t)
    stemmedSentence = []
    ps = PorterStemmer()
    for word in tokenisedWords:
        stemmedSentence.append(ps.stem(word))  # stem word
        stemmedSentence.append(" ")
    return "".join(stemmedSentence)

# Graphs number of labels per question
def graph_labels():
    # Graph of Labels
    temp = {}
    for data in sorted(num_tags_per_question):
        print('Questions with ', data, ' Tag(s) -> ', num_tags_per_question[data])
        temp[data] = num_tags_per_question[data]
    plot.plot(list(temp.keys()), list(temp.values()), color='red', linestyle='dashed',
              linewidth=3, marker='o', markerfacecolor='blue', markersize=12)
    plot.xlabel('Number of Tags')
    plot.ylabel('Occurrences')
    plot.title('Number of Times n Labels Were Attached to a Question')
    plot.show()

# Graph top 5 frequent tags
def graph_top_tags():
    # Graph of top 5 tags
    print("\nTop 5 Tags: ")
    sort_tags = sorted(tagCounts.items(), key=lambda x: x[1], reverse=True)
    tag = []
    total = []
    c = 0
    for data in sort_tags:
        print('Tag: ', data[0], "\tFrequency: ", data[1])
        tag.append(data[0])
        total.append(data[1])
        if c == 4:
            break
        else:
            c = c + 1
    plot.bar(range(len(tag)), total, align='center', color=['blue', 'red'])
    plot.xticks(range(len(tag)), tag)
    plot.xlabel('Tag')
    plot.ylabel('Frequency')
    plot.title('Most Frequently Used Tags')
    plot.show()

# Train Test and call evaluate method
def train_test_evaluate_model(xtrain, ytrain, xtest, ytest, classifier):
    # Pipeline with TF-IDF for feature extraction and OvsR for classification with classifier
    pipeline = Pipeline([
        ('Tf-IDF', TfidfVectorizer()),
        ('Classifier', OneVsRestClassifier(classifier))
    ])
    # Fit the data
    pipeline.fit(xtrain, ytrain)
    # Predict on test
    predictions = pipeline.predict(xtest)
    # Evaluate performance
    evaluate_classifier(ytest, predictions)

# Evaluation Metrics
def evaluate_classifier(ground_truth, pred):
    accuracy = accuracy_score(ground_truth, pred) * 100
    recall = recall_score(ground_truth, pred, average='micro') * 100
    precision = precision_score(ground_truth, pred, average='micro', zero_division=1) * 100
    f1 = f1_score(ground_truth, pred, average='micro', zero_division=1) * 100
    hamming = hamming_loss(ground_truth, pred) * 100
    print('F1 Score =     ', round(f1, 2), '%')
    print('Accuracy =     ', round(accuracy, 2), "%")
    print('Recall =       ', round(recall, 2), '%')
    print('Hamming Loss = ', round(hamming, 2), '%')
    print('Precision =    ', round(precision, 2), '%\n')


# =========================================================================================
# ================================Implementation===========================================
# =========================================================================================

# Get a data frame with Question ad tags column
df = create_data_frame()

# Print initial Data frame
print("Data Frame Before Binarization:")
print(df)

# Show Graphs
graph_labels()
graph_top_tags()

# Initialize the MLB
mlb = MultiLabelBinarizer()
binarizered_result = mlb.fit_transform([str(df.loc[i, 'Tags']).split(' ') for i in range(len(df))])
df = pd.concat([df['Question'], pd.DataFrame(binarizered_result, columns=list(mlb.classes_))], axis=1)

# Get all the Programming Tags
train_tags = []
data_frame_cols = list(df.columns)
for col in data_frame_cols:
    if col != 'Question':
        train_tags.append(col)

# Split the Data using Sci-Kit Learns method
X_train, X_test, y_train, y_test = train_test_split(df["Question"], df[train_tags], test_size=0.2, random_state=42)

# Number of rows for Training
print('\nNumber of points in Training Data: ', X_train.shape[0])

# Number of rows for testing
print('Number of points in Test Data: ', y_test.shape[0], '\n')

# Classifiers
support_vector_classifier = LinearSVC()
multinomial_bayes = MultinomialNB(fit_prior=True, class_prior=None)

# Train Test and Evaluate LinearSVC Model
print('\nSupport Vector Classifier: ')
train_test_evaluate_model(X_train, y_train, X_test, y_test, support_vector_classifier)

# Train Test and Evaluate Multinomial Bayes Model
print("Multinomial Bayes: ")
train_test_evaluate_model(X_train, y_train, X_test, y_test, multinomial_bayes)