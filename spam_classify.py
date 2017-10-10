import os	#created and improvised by Aditya Mohan #7avenged
import io   #THIS IS JUST ONE WAY TO LOAD FILES IN VARIABLE FOR TRAINING, MANY OTHER WAYS ARE ALSO POSSIBLE
import numpy                                 
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB      #THIS SHALL IMPLEMENT NAIVE BAYES

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})
#THESE COMMANDS WILL LOAD UP THE DATA(EMAILS OF SPAM AND NON-SPAM IN THE 2 DIFFERENT CLASSES OF THE data VARIABLE,FOR TRAINING
data = data.append(dataFrameFromDirectory('DIRECTORY CONTAINING THE SPAM DATASET FOLDER/spam', 'A SPAM!!!'))
data = data.append(dataFrameFromDirectory('DIRECTORY CONTAINING THE NON-SPAM DATASET FOLDER/non-spam', 'Nope!Not a spam !'))

#THIS COMMAND WILL TOKENIZE OR CONVERT ALL THE VALUES IN THE COLUMN "MESSAGE" OF THE data VARIABLE 
#ie. ASSIGN A NUMERICAL VALUE TO EACH INDIVIDUAL WORD IN THE MESSAGE VARIABLE
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values) #SO, IT WILL RECORD HOW MANY TIMES EACH WORD OCCURSIN THE EMAIL


classifier = MultinomialNB()
targets = data['class'].values #THIS IS THE CLASSIFICATION DATA FOR EACH EMAIL
classifier.fit(counts, targets) #MODEL HAS BEN CREATED

#NOTE:- NO TEST SET HERE AS I USED SAMPLE TEXT (AFTER CONVERTING IT TO VECTORS  ETC ETC.) FOR TESTING PURPOSE 
examples = ['Make 5000 Rs. per day!!!, unlock this hidden trick', "Hello Aditya, I am making a team for a hackathon, you interested? Ping me up. "]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions

