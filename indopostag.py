from nltk import word_tokenize, pos_tag
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import pickle

"""
    Training data handler
"""

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def tokenize(file_name):

    sentences = []
    word_tag = []
    first_sentence = True

    with open (file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if (line[0] != '\n' and line[0] != '#'):
                tokens = line.split('\t')
                if (first_sentence):
                    first_sentence = False
                elif (line [0:2] == '1\t' and not first_sentence):
                    sentences.append(word_tag)
                    word_tag = []
                word_tag.append((tokens[1], tokens[3]))

        if (len(word_tag) != 0):
            sentences.append(word_tag)

    return sentences

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    x, y = [], []
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            x.append(features(untag(tagged), index))
            y.append(tagged[index][1])
    return x, y

tagged_sentences = tokenize('corpus/id-ud-dev.conllu')
x, y = transform_to_dataset(tagged_sentences)

# Split the dataset for training and testing
cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print len(training_sentences)
print len(test_sentences)

clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', DecisionTreeClassifier(criterion='entropy'))])
clf.fit(x[:10000], y[:10000])
print 'Training completed'
x_test, y_test = transform_to_dataset(test_sentences)
print "Accuracy:", clf.score(x_test, y_test)
