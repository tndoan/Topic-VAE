import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def isDigit(s):
    try:
        float(s)
        return True
    except:
        return False

def getVocabList(fname, vocabName):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    vocabSet = set()
    # get vocabulary list
    for line in lines:
        text = line.decode('unicode-escape').encode('ascii', 'ignore')
        tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
        for token in filter(lambda word: word not in ',.', tokens):
            if not isDigit(token):
                if not token in string.punctuation:
                    vocabSet.add(token.lower())
    f = open(vocabName, 'w')
    for voc in vocabSet:
        f.write(voc + '\n')
    f.close()

def extractFeature(fname, vocabDict, swList, output):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    f = open(output, 'w')
    for line in lines:
        text = line.decode('unicode-escape').encode('ascii', 'ignore')
        sentences = sent_tokenize(text)
        eachDoc = ''
        for sent in sentences:
            tokens = [word for word in word_tokenize(sent)]
            for token in filter(lambda word: word not in ',.', tokens):
                if not isDigit(token):
                    if not token in string.punctuation:
                        idx = vocabDict.get(token.lower())
                        if token.lower() in swList:
                            isSW = 1
                        else:
                            isSW = 0
                        if idx == None:
                            continue
                        eachDoc += str(idx) + ':' + str(isSW) + ' '
            eachDoc += '. '
        f.write(eachDoc + '\n')
    f.close()

def loadVoc(vocabFName):
    """return a dict whose key is word and value is the number which represents
    this word"""
    f = open(vocabFName, 'r')
    lines = f.readlines()
    f.close()
    result = dict()
    for idx, val in enumerate(lines):
        word = val.strip('\n')
        result[word] = idx
    return result
        
def loadStopWordList(swFile):
    """load  the list of stop words"""
    f = open(swFile, 'r')
    lines = f.readlines()
    f.close()
    result = list()
    for line in lines:
        sWord = line.strip('\n')
        result.append(sWord)
    return result

if __name__ == '__main__':
    #getVocabList('../../raw/abs_test.txt', 'vocab_test.txt')
    vocabDict = loadVoc('../../raw/vocab_test.txt') 
    swList = loadStopWordList('../../raw/stopwords.txt')
    extractFeature('../../raw/abs_test.txt', vocabDict, swList, '../../raw/featureTest.txt')
