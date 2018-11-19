
def readData(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    result = list()
    for line in lines:
        eachDoc = list()
        sentences = line.strip('\n').split('.')
        for sentence in sentences:
            comp = sentence.split(' ')
            eachSentence = list()
            for c in comp:
                if c == '':
                    continue
                w = c.split(':')
                eachSentence.append((int(w[0]), int(w[1])))
            if len(eachSentence) != 0:
                eachDoc.append(eachSentence)
        result.append(eachDoc)
    return result

if __name__ == '__main__':
    print readData('../../data/featureTest.txt')
