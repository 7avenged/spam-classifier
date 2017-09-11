import os	#created and improvised by Aditya Mohan #7avenged
import io
import numpy  
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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

data = data.append(dataFrameFromDirectory('DIRECTORY CONTAINING THE SPAM DATASET FOLDER/spam', 'A SPAM!!!'))
data = data.append(dataFrameFromDirectory('DIRECTORY CONTAINING THE NON-SPAM DATASET FOLDER/non-spam', 'Nope!Not a spam !'))


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)


examples = ['Make 5000 Rs. per day!!!, unlock this hidden trick', "Hello Aditya, I am making a team for a hackathon, you interested? Ping me up. "]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions

