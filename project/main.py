import nltk
import numpy as py
from nltk.corpus import stopwords
from nltk.tree import *
from nltk.corpus import wordnet as wn
from difflib import SequenceMatcher
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


stop_words=set(stopwords.words('english'))


def readfile(addr):
	f=open(addr,'r')
	raw=f.read()
	return raw

def token(raw):
	sentlist=[]
	sents=nltk.sent_tokenize(raw)
	for sent in sents:
		tokenw=nltk.word_tokenize(sent)
		tagw=nltk.pos_tag(tokenw)
		sentlist.append(tagw)
	return sentlist


def opinionS(sent):
	ow=False
	adj=False
	for word in sent:
		if word[1]=='NN' or word[1]=='NNS' or word[1]=='NNP' or word[1]=='NNPS':
			ow=True
		if word[1]=='JJ' or word[1]=='JJR'or word[1]=='JJS':
			adj=True
	return ow*adj

def NounPhrase(sent):
	feature=[]
	grammer = r"""
		NP:
			{<NN|NNS><NN|NNS><NN|NNS>}
			{<NN|NNS><NN|NNS>}
			{<NN|NNS><IN><NN|NNS><NN|NNS>}
			{<NN|NNS><IN><NN|NNS>}
			{<NN|NNS>}
	"""
	cp=nltk.RegexpParser(grammer)
	result = cp.parse(sent)
	for subtree in result.subtrees(filter = lambda t:t.label() == 'NP'):
		feature.append (subtree.leaves())
	return feature

positive=['good','pretty','fantastic','cool','nice','amazing','excellent','perfect','outstanding','clear','remarkable','gorgeous','wonderful','awesome','upbeat','favorable','cheerful','pleased','appealing']
negative=['bad','disappointing','dull','ugly','terrible','disgraceful','poor','shoddy','awful','noisome','disgusting','frustrating','awkward','irritating','weired']
seed_list = {}
for word in positive:
	seed_list[word] = 'positive'
for word in negative:
	seed_list[word] = 'negative'

negation_word = ["no","not","yet","never","hardly","little","few","none"]

print("(1) Apex AD2600 Progressive-scan DVD player cleaned.txt\n")
print("(2) Canon G3 cleaned.txt\n")
print("(3) Creative Labs Nomad Jukebox Zen Xtra 40GB cleaned.txt\n")
print("(4) Nikon coolpix 4300 cleaned.txt\n")
print("(5) Nokia 6610 cleaned.txt\n")

val = input("Enter file number you wish to process: ")


if val == '1':
	file_name = 'Apex AD2600 Progressive-scan DVD player cleaned.txt'
elif val == '2':
	file_name = 'Canon G3 cleaned.txt'
elif val == '3':
	file_name = 'Creative Labs Nomad Jukebox Zen Xtra 40GB cleaned.txt'
elif val == '4':
	file_name = 'Nikon coolpix 4300 cleaned.txt'	
elif val == '5':
	file_name = 'Nokia 6610 cleaned.txt'
else:
	raise Exception('input should be 1-5. The value of input was: {}'.format(val))



file = 'data/' + file_name

print("Star calculating results...\n")

raw=readfile(file)
tokenized=token(raw)
OS=[sent for sent in tokenized if opinionS(sent)]
opinionS_N=len(OS)
nounphrase=[]
for sent in OS:
	nounphrase.append(NounPhrase(sent))
nounphrase_N=len(nounphrase)
candidate=[]
for i in range(0,nounphrase_N):
	for j in range(0,len(nounphrase[i])):
		f=''
		for x in range(0,len(nounphrase[i][j])):
			if (nounphrase[i][j][x][0] not in stop_words or nounphrase[i][j][x][0] in ['of','for']) and (x!=len(nounphrase[i][j])-1):
				f+=nounphrase[i][j][x][0]+' '
			elif(nounphrase[i][j][x][0] not in stop_words or nounphrase[i][j][x][0] in ['of','for']) and (x==len(nounphrase[i][j])-1):
				f+=nounphrase[i][j][x][0]
		candidate.append(f)
candidate=[elem for elem in candidate if elem.strip()]
candidateDic={}
for i in candidate:
	if i not in candidateDic:
		candidateDic[i]=1
	else:
		candidateDic[i]+=1

features=[elem for elem in candidateDic.keys() if candidateDic[elem]/opinionS_N > 0.02]



featuresNS=[]
for f in features:
	s=f
	s=s.replace(' ','')
	featuresNS.append(s)

featuresDic={}

for i in range(0,len(OS)):
	sentNS=''
	sent_N=len(OS[i])
	for j in range(0,sent_N):
		sentNS+=OS[i][j][0]
	
	for z in range(0,len(featuresNS)):
		if featuresNS[z] in sentNS:
			if features[z] not in featuresDic:
				featuresDic[features[z]]=[i]
			else:
				featuresDic[features[z]].append(i)

def remove_tag(OS):
	output = []
	for sent in OS:
		new_sent = []
		for word in sent:
			new_sent.append(word[0])
		output.append(new_sent)
	return output

OS_notag = remove_tag(OS)

def find_syn_ant(word):
	synonyms = []
	antonyms = []
	for syn in wn.synsets(word):
		for l in syn.lemmas():
			synonyms.append(l.name())
			if l.antonyms():
				antonyms.append(l.antonyms()[0].name())

	return synonyms, antonyms

def negation(orientation):
	if orientation == "positive":
		orientation = "negative"
	else:
		orientation = "positive"
	return orientation

def OrientationPrediction(adj_list, seed_list):
	while True:
		size1 = len(seed_list)
		adj_list, seed_list = OrientationSearch(adj_list, seed_list)
		size2 = len(seed_list)
		if size1 == size2:
			break

	return adj_list, seed_list

def OrientationSearch(adj_list, seed_list):
	added = False
	for adj in adj_list:
		adj_syn, adj_ant = find_syn_ant(adj)
		for syn in adj_syn:
			if syn in seed_list:
				adj_orientation = seed_list[syn]
				seed_list[adj] = adj_orientation
				added = True
				break
		if added == False:
			for ant in adj_ant:
				if ant in seed_list:
					adj_orientation = negation(seed_list[ant])
					seed_list[adj] = adj_orientation
					added = True
					break

	return adj_list, seed_list

def close_word(word, sentence, size):
	word_pos = sentence.index(word)
	if len(sentence) <= size:
		window = sentence
	elif word_pos < size:
		window = sentence[0:word_pos + size]
	elif len(sentence) - word_pos < size:
		window = sentence[word_pos - size:-1]
	else:
		window = sentence[word_pos - size: word_pos + size]

	return window


for feature in featuresDic:
	for sentence_index in featuresDic[feature]:
		sentence = OS_notag[sentence_index]
		if feature in sentence:

			window = sentence
			
			adjs = []
			window_tag = nltk.pos_tag(window)
			for word_tag in window_tag:
				if word_tag[1] == 'JJ':
					adjs.append(word_tag[0])

			adjs, seed_list = OrientationPrediction(adjs,seed_list)




def wordOrientation(word, sentence):
	orientation = seed_list[word]
	window = close_word(word, sentence, 5)
	for neg_word in negation_word:
		if neg_word in window:
			orientation = negation(orientation)

	if orientation == "positive":
		return 1
	else:
		return -1

sentenceOrientation = {}
sentence_effective = {}
sentence_opw = {}
sentence_feature = {}

for i,sentence in enumerate(OS_notag):
	orientation = 0
	sentence_opw[i] = []
	sentence_effective[i] = []
	sentence_feature[i] = []

	for feature in featuresDic:
		if feature in sentence:
			sentence_feature[i].append(feature)

			eff_window = close_word(feature, sentence, 5)
			eff_tag = nltk.pos_tag(eff_window)
			for tag in eff_tag:
				if tag[1] == 'JJ' and tag[0] not in sentence_effective[i]:
					sentence_effective[i].append(tag[0])

	for word in sentence:
		if word in seed_list:
			sentence_opw[i].append(word)

	for op in sentence_opw[i]:
		if op in seed_list:
			orientation += wordOrientation(op,sentence)

	if orientation > 0:
		sentenceOrientation[i] = "Positive"
	elif orientation < 0:
		sentenceOrientation[i] = "Negative"
	else:
		for eff_op in sentence_effective[i]:
			if eff_op in seed_list:
				orientation += wordOrientation(eff_op,sentence)
		if orientation > 0:
			sentenceOrientation[i] = "Positive"
		elif orientation < 0:
			sentenceOrientation[i] = "Negative"
		else:
			sentenceOrientation[i] = "Neutral"

featureOrientation = {}

for feature in featuresDic:
	featureOrientation[feature] = {"positive":[], "negative":[], "neutral":[]}

	for sentence_index in featuresDic[feature]:

		if sentenceOrientation[sentence_index] == "Positive" and sentence_index not in featureOrientation[feature]["positive"]:
			featureOrientation[feature]["positive"].append(sentence_index)

		elif sentenceOrientation[sentence_index] == "Negative" and sentence_index not in featureOrientation[feature]["negative"]:
			featureOrientation[feature]["negative"].append(sentence_index)

		elif sentenceOrientation[sentence_index] == "Neutral" and sentence_index not in featureOrientation[feature]["neutral"]:
			featureOrientation[feature]["neutral"].append(sentence_index)


def merge_two_dicts(x, y):
	z = {"positive":[], "negative":[], "neutral":[]}
	for key in z.keys():
		z[key] = x[key] + y[key]
	return z

duplicate_feature = []
for i,prev_feature in enumerate(features):
	for feature in features[i+1:]:
		s = SequenceMatcher(None, prev_feature, feature)
		if s.ratio() > 0.7 and s.ratio() != 1.0:
			featureOrientation[feature] = merge_two_dicts(featureOrientation[prev_feature],featureOrientation[feature])
			duplicate_feature.append(prev_feature)

for feature in duplicate_feature:
	del featureOrientation[feature]



def list_sentence(input):
	return [[' '.join(i)] for i in input]

sentences = list_sentence(OS_notag)

output_file = 'output/' + file_name.replace(' cleaned','_output')
print("Start outputing results to " + output_file + '\n')

output = open(output_file,'w')
# print output loop
for feature in featureOrientation:
	if featureOrientation[feature]["positive"] != [] and featureOrientation[feature]["negative"] != []:
		output.write(feature + '\n')
		if featureOrientation[feature]["positive"] != []:
			output.write("Positive:" + '\n')
			for index in featureOrientation[feature]["positive"]:
				output.write(sentences[index][0].replace("#", "").strip(" ") + '\n')

		if featureOrientation[feature]["negative"] != []:
			output.write("Negative:"+ '\n')
			for index in featureOrientation[feature]["negative"]:
				output.write(sentences[index][0].replace("#", "").strip(" ") + '\n')

		if featureOrientation[feature]["neutral"] != []:
			output.write("neutral:"+ '\n')
			for index in featureOrientation[feature]["neutral"]:
				output.write(sentences[index][0].replace("#", "").strip(" ") + '\n')

output.close()

print("Output completes" + '\n')

val = input("Do you want to evaluate the output?(y/n): ")

if val.lower() == 'y':
	print("Start evaluating output results ..." + '\n')
	# evaluation
	eval_list = []
	for feature in featureOrientation:

		for sentence_index in featureOrientation[feature]["positive"]:
			if sentence_index not in eval_list:
				eval_list.append(sentence_index)

		for sentence_index in featureOrientation[feature]["negative"]:
			if sentence_index not in eval_list:
				eval_list.append(sentence_index)

		for sentence_index in featureOrientation[feature]["neutral"]:
			if sentence_index not in eval_list:
				eval_list.append(sentence_index)


	eval_list.sort()


	addr = file.replace(' cleaned','')
	f = open(addr,'r')
	raw = f.read()
	add = False

	in_sentence = ''
	for i,char in enumerate(raw):
		if char == '#' and raw[i - 1] == ']':
			add = True
		if add:
			in_sentence += raw[i]
		if raw[i] == '.' or raw[i] == '!':
			add = False

	f.close()

	exist_eval_list = in_sentence.split('##')
	exist_eval_list.remove('')

	total_op = len(exist_eval_list)
	total_correct = 0

	for index in eval_list:
		output_s = sentences[index][0].replace("#", "").strip(" ")
		for sentence in exist_eval_list:
			s = SequenceMatcher(None, output_s, sentence)
			if s.ratio() > 0.9:
				total_correct += 1
				break

	print("Sentence orientation accuracy is:")
	print("%.3f" % (total_correct / total_op))
	print('\n')

	total_correct = 0
	for sentence in sentences:
		sentence = sentence[0].replace("#", "").strip(" ")
		for sentence_comp in exist_eval_list:
			s = SequenceMatcher(None, sentence, sentence_comp)
			if s.ratio() > 0.9:
				total_correct += 1
				break


	print("Opinion sentence extraction precision is:")
	print("%.3f" % (total_correct / total_op))
	print('\n')

	print("Program finished")

elif val.lower() == 'n':
	print("Program finished")

else:
	raise Exception('input should be y/n. The value of input was: {}'.format(val))




