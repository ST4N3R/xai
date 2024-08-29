import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup


class TextPreprocessing:
    def __init__(self) -> None:
        self.datastore = None
        self.sentences = []
        self.labels = []


    #Loading data to the TextProcesssing instance
    def set_datastore(self, datastore) -> None:
        self.datastore = datastore

    
    #Get datastore
    def get_datastore(self) -> list:
        return self.datastore


    #Remove interpunctation signs from sentences
    def remove_interpunctation(self, item: dict) -> str:
        sentence = item['headline'].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        return sentence


    #Remove HTML signs and split sentence into list
    def get_words(self, sentence: str) -> list:
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        return sentence.split()


    #Filter uninformative words
    def filter_word(self, word: str, filtered: str) -> str:
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]
        
        table = str.maketrans('', '', string.punctuation)

        word = word.translate(table)
        if word not in stopwords:
            filtered = filtered + word + " "
        return filtered


    #Clean data from interpuntation, uninformative words and HTML signs
    def clean_datastore(self) -> None:
        if self.datastore == None:
            print("Load data first!")
            return None
        
        urls = []
        for item in self.datastore:
            sentence = self.remove_interpunctation(item=item)
            words = self.get_words(sentence=sentence)

            filtered_sentence = ""
            for word in words:
                filtered_sentence = self.filter_word(word=word, filtered=filtered_sentence)

            self.sentences.append(filtered_sentence)
            self.labels.append(item['is_sarcastic'])
            urls.append(item['article_link'])

        return None


    #Split sentences into train and test
    def split_sentences(self, training_size) -> None:
        self.X_train = self.sentences[0:training_size]
        self.X_test = self.sentences[training_size:]
        self.y_train = self.labels[0:training_size]
        self.y_test = self.labels[training_size:]


    #Create sentences and pad them
    def create_sequences(self, data, max_lenght, padding_type, trunc_type):
        sequences = self.tokenizer.texts_to_sequences(data)
        padded = pad_sequences(sequences, maxlen=max_lenght, padding=padding_type, truncating=trunc_type)
        return padded


    #Initilize tokenizer object and create padded senteces
    def initize_tokenizer(self, vocab_size=8_000, max_length=85, trunc_type='post', padding_type='post', oov_tok="", training_size=23_000):
        self.split_sentences(training_size=training_size)

        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        self.tokenizer.fit_on_texts(self.X_train)

        self.X_train_padded = self.create_sequences(data=self.X_train, max_lenght=max_length, padding_type=padding_type, trunc_type=trunc_type)
        self.X_test_padded = self.create_sequences(data=self.X_test, max_lenght=max_length, padding_type=padding_type, trunc_type=trunc_type)

    
    #Get train and test data
    def get_data(self, asNpList=True, asPadded=True):
        if asNpList:
            self.X_train_padded = np.array(self.X_train_padded)
            self.X_test_padded = np.array(self.X_test_padded)
            self.y_train = np.array(self.y_train)
            self.y_test = np.array(self.y_test)
        
        if asPadded:
            return self.X_train_padded, self.X_test_padded, self.y_train, self.y_test
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test