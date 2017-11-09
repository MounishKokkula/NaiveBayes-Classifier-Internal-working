import nltk as nl
import math

f = open('NaiveBayesTestInput.txt','r')

listOfDocs = []
listOfClasses = []

for l in f.readlines():
    listOfDocs.append(l.split("-")[0])
    if l.split("-")[1].replace('\n', '') not in listOfClasses:
        listOfClasses.append(l.split("-")[1].replace('\n', ''))
f.close()
# print("listOfClasses "+ str(listOfClasses))

def extractVocab(listOfDocs):
    vocabulary = []
    for d in listOfDocs:
        vocabulary.extend(nl.word_tokenize(d))
    return vocabulary

def concattxt(c):
    docsInC = []
    f = open('NaiveBayesTestInput.txt', 'r')
    # print("1.printing c " + str(c))
    for l in f.readlines():
        rawTestDoc = l.split("-")
        # print("2. inside 2nd for " + str(rawTestDoc))
        if rawTestDoc[1].replace('\n', '') == c.replace('\n', ''):
            # print("3.Inside if loop " + str(docsInC.extend(nl.word_tokenize(rawTestDoc[0]))))
            docsInC.extend(nl.word_tokenize(rawTestDoc[0]))

    f.close()
    return docsInC


def counttokens(doc, j):
    count = 0
    for term in doc:
        if term.replace('\n', '') == j.replace('\n', ''):
            count = count + 1
    return count

def countTotalTokens(doc, j):
    count = 0
    for term in doc:
        if term.replace('\n', '') != j.replace('\n', ''):
            count = count + 1
    return count


def TrainMultinomialNB(C, D):
    # V ← EXTRACTVOCABULARY(D)
    v = extractVocab(D)
    #  N ← COUNTDOCS(D)
    n = D.__len__()
    prior = dict()
    text = dict()
    condprob = {}
    condpro = {}
    docsInC = {}
    tCount = 0
    totCount = 0

    for t in v:
        for c in C:
            condprob[t] = {}
            condprob[t][c] = {}

    for c in C:
        # print(c)
        f = open('NaiveBayesTestInput.txt', 'r')
        docsInC[c.replace('\n', '')] = []
        # print("1.printing c " + str(c))
        for l in f.readlines():
            rawTestDoc = l.split("-")
            # print("2. inside 2nd for " + str(rawTestDoc))
            if rawTestDoc[1].replace('\n', '') == c.replace('\n', ''):
                docsInC[c.replace('\n', '')].append(rawTestDoc[0])
        f.close()

        # Nc ← COUNTDOCSINCLASS(D, c)
        Nc = docsInC[c].__len__()

        # prior[c] ← Nc/N
        prior[c] = Nc / n

        # textc ← CONCATENATETEXTOFALLDOCSINCLASS(D, c)
        text[c] = concattxt(c)

        totCount = len(text[c])

        # for each t ∈ V
        # do Tct ← COUNTTOKENSOFTERM(textc, t)
        # do condprob[t][c] ← Tct+1/ ∑t′(Tct′+1)
        for t in v:
            tCount = counttokens(text[c], t)
            condprob[t][c] = ((tCount + 1) / (totCount + 1))

    return v, prior, condprob


# APPLYMULTINOMIALNB(C, V, prior, condprob, d)
# 1 W ← EXTRACTTOKENSFROMDOC(V, d)

def ApplyMultinomialNB(C, V, prior, condprob, D):
    w = nl.word_tokenize(D)

    score = dict()
    for c in C:
        score[c] = math.log(prior[c])
        for t in w:
            score[c] += math.log(condprob[t][c])
    return score.items()



# print(TrainMultinomialNB(listOfClasses, listOfDocs))
trainingData = TrainMultinomialNB(listOfClasses, listOfDocs)
# print(math.log(trainingData[1]["Y"]))
testDoc="David Wright and the Mets"
print(ApplyMultinomialNB(listOfClasses,trainingData[0],trainingData[1],trainingData[2],testDoc))