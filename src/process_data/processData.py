import string, glob, re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def writeToFile(lOfDocuments, fname):
    """
    """
    f = open(fname, 'w')
    for doc in lOfDocuments:
        stringOfDoc = ''
        for sentence in doc:
            for (idx, isSW) in sentence:
                stringOfDoc += str(idx) + ':' + str(isSW)
            stringOfDoc += '.'
        f.write(stringOfDoc + '\n')
    f.close()

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"<br />",r" ",s)
    # s = re.sub(' +',' ',s)
    s = re.sub(r'(\W)(?=\1)', '', s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def loadFile(fname, vocDict, swList):
    """read the content of the file and based on vocabulary dictionary to get
    the index of each word and indicate if the word is stop word or not.
    Since we care about the order of words, we use list to store. Each sentence
    is a list and whole document is the list of list. """
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    result = list()
    punc = string.punctuation
    punc = punc.replace("-", "").replace("'", "")
    errorWords = list()
    for line in lines:
        sentences = sent_tokenize(line.decode('unicode-escape').encode('ascii', 'ignore'))
        #sentences = line.split('.')
        for sentence in sentences:
            #words = [i for i in normalizeString(sentence).strip().split(' ')]
            words = [i.lower() for i in re.split("[" + punc + " ]+", sentence)]
            #words = [i.lower() for i in word_tokenize(sentence) if i not in punc]
            sResult = list()
            for word in words:
                if word == '' or word.isdigit():
                    continue
                if word.startswith("'") or word.startswith("-"):
                    word = word[1:]
                elif word[-1] == "'":
                    word = word[:-1]
                elif word[-2:] == "'s":
                    word = word[:-2]
                idx = vocDict.get(word)
                if idx == None and word != ".":
                    errorWords.append(word)
                isSW = word in swList
                if isSW:
                    i = 1
                else:
                    i = 0
                sResult.append((str(idx), str(i)))
            if len(sResult) != 0:
                result.append(sResult)
    if len(errorWords) != 0:
        print fname
        print errorWords
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

if __name__ == "__main__":
    d = loadVoc('../../raw/imdb.vocab')
    swList = loadStopWordList('../../raw/stopwords.txt')
    print loadFile('../../raw/train/pos/2_9.txt', d, swList)
    lOfFiles = glob.glob('../../raw/train/pos/*')
    result = list()
    for fname in lOfFiles:
        result.append(loadFile(fname, d, swList))
    #writeToFile(result, '../../raw/train.txt')
