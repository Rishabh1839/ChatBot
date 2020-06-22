# Rishabh Singh
# MS548
# University of Advancing Technology
#---------------------------------------------------------------------------------------
# Chatbot implementation
# REFERENCE: https://python.gotrained.com/chatbot-development-python-nltk/
#---------------------------------------------------------------------------------------
# importing necessary libraries for development of the chatbot
import bs4 as bs
# the urllib will help us make a connection to a remote website
import urllib.request
# this will help us perform regex operation
import re
# nltk is used for natural language processing
import nltk
# numpy for basic array operations
import numpy as np
# random number generation
import random
# used for string manipulation
import string

# Now we will be scraping and preprocessing a wikipedia article
# The bot will be anle to answer questions related with the article
# And we will be using urllib in order to pull that webpage and have our bot learn from it
raw_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/COVID-19_pandemic')
raw_data = raw_data.read()

# this will convert it into an HTML format since default is binary for the urlopen function and lxml parameters
html_data = bs.BeautifulSoup(raw_data, 'lxml')

# the find_all method allows us to extract the paragraphs
# we need to pass p as parameter to the method which stands for paragraphs
paragraphs = html_data.find_all('p')
articleContent = ""

# convertss the final text into lowercase
for p in paragraphs:
    articleContent += p.text
articleContent = articleContent.lower()

# This will remove numebrs from our dataset and replace it with empty spaces with single space
articleContent = re.sub(r'\[[0-9]*\]', ' ', articleContent)
articleContent = re.sub(r'\s+', ' ', articleContent)

# Next step is to tokenize from sentences to words retrieved from article
sentenceList = nltk.sent_tokenize(articleContent)
# this one tokenizes the article into words
articleWords = nltk.word_tokenize(articleContent)

# Lemmatization helps find similarity between words since the similar words can be
# used in different tense and degrees, using this will make them uniform
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatization(words):
    return [lemmatizer.lemmatize(word) for word in words]
removePunctuation = dict((ord(punctuation), None) for punctuation in string.punctuation)

# this accepts a text string and performs a lemmatization on the string by passing it to
# lemmatize words function which lemmatizes the words. Punctuations are also removed from text
def RemovePunctuations(text):
    return lemmatization(nltk.word_tokenize(text.lower().translate(removePunctuation)))

# As the website we will retrieve information doest contain any greeting messages
# we will create a function to greet the user by creating two lists with different messages
# The user input will be checked against the words in one of the greetings lists
greeting01 = ("hey", "heys", "hello", "morning", "evening", "greetings",)
greeting02 = ["hey", "hey hows you?", "*nods*", "hello there", "ello", "Welcome, how are you"]

def respond(text):
    for word in text.split():
        if word.lower() in greeting01:
            return random.choice(greeting02)

# we create a method for general response generation
# we will use TF-IDF approach in order to conver word to vectors
# So we can convert our words to vectors or numbers and then apply cosine similarity to find similar vectors
# we can use TFifvectorizer module from sklearn to convert words to their TF-IDF counterparts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# this function is used for response generation
def reply(user_input):
    chatbot_response = ''
    sentenceList.append(user_input)
    word_vectors = TfidfVectorizer(tokenizer = RemovePunctuations, stop_words = 'english')
    vectorized_words = word_vectors.fit_transform(sentenceList)
    similarity_values = cosine_similarity(vectorized_words[-1], vectorized_words)
    similar_sentence_number = similarity_values.argsort()[0][-2]
    similar_vectors = similarity_values.flatten()
    similar_vectors.sort()
    matched_vector = similar_vectors[-2]

    if(matched_vector == 0):
        chatbot_response = chatbot_response + "I apologize, I didn't catch that."
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sentenceList[similar_sentence_number]
        return chatbot_response
# set a flag to true
continue_discussion = True
print("Hello user, I am a wiki chatbot and I'll be answering your questions regarding COVID-19:")
# we execute a while loop inside where we ask the user to input their questions
while(continue_discussion == True):
    user_input = input()
    user_input = user_input.lower()
    # This loop will terminate until user inputs 'bye'
    if(user_input != 'bye'):
        # if the user says thanks, thank you or thank you very much the bot will respond with
        # Most welcome
        if(user_input == 'thanks' or user_input == 'thank you very much' or user_input == 'thank you'):
            continue_discussion = False
            print("Chatbot: Most Welcome")
        else:
            if(respond(user_input)!=None):
                print("Chatbot: " + respond(user_input))
            else:
                print("Chatbot: ", end="")
                print(reply(user_input))
                sentenceList.remove(user_input)
    else:
        continue_discussion = False
        print("Chatbot: My pleasure, bye ...")
