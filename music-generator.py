import math
import numpy as np
import pyaudio as pa
import matplotlib.pyplot as plt
import os
import time
import sys
import json

START_TIME = time.time()

#CONSTANTS
keySigs = {		'A':0,
		 		'A-sharp':1,
		 		'B-flat':1,
		 		'B':2,
		 		'C-flat':2,
		 		'C':3,
		 		'C-sharp':4,
		 		'D-flat':4,
		 		'D':5,
		 		'D-sharp':6,
		 		'E-flat':6,
		 		'E':7,
		 		'F':8,
		 		'F-sharp':9,
		 		'G-flat':9,
		 		'G':10,
		 		'G-sharp':11,
		 		'A-flat':11,	}

inverseKeySigs = {}

SCALES = {
'major':[10,0,2,0,3,4,0,5,0,2,0,3],
'majorBin':[1,0,1,0,1,1,0,1,0,1,0,1],
'minor':[10,0,2,3,0,4,0,5,2,0,2,0],
'minorBin':[1,0,1,1,0,1,0,1,1,0,1,0]}

CHORDS = {'M':[1,0,0,0,1,0,0,1,0,0,0,0],
		  'm':[1,0,0,1,0,0,0,1,0,0,0,0],
		  '7':[1,0,0,0,1,0,0,1,0,0,1,0],
		  'maj7':[1,0,0,0,1,0,0,1,0,0,0,1],
		  'min7':[1,0,0,1,0,0,0,1,0,0,1,0],
		  'dim':[1,0,0,1,0,0,1,0,0,0,0,0],
		  'dim7':[1,0,0,1,0,0,1,0,0,1,0,0],
		  'aug':[1,0,0,0,1,0,0,0,1,0,0,0]}

CHORD_COMBOS = {'major':{'0':['M'],
						 '1':[],
						 '2':['m'],
						 '3':[],
						 '4':['m'],
						 '5':['M'],
						 '6':[],
						 '7':['M'],
						 '8':[],
						 '9':['m'],
						 '10':[],
						 '11':['dim']},
				'minor':{'0':['m'],
						 '1':[],
						 '2':['dim'],
						 '3':['M'],
						 '4':[],
						 '5':['m'],
						 '6':[],
						 '7':['m','7'],
						 '8':['M'],
						 '9':[],
						 '10':['M'],
						 '11':[]}}

SPECTRA = {}
DECAYS = {}
#END CONSTANTS

#BEGIN DEPENDENT VARIABLES
notesPerMeasure = -1
notesPerPiece = -1
DURATION = -1
FREQS = []
FREQ_LETTERS = []
PROX_BASED_WEIGHTS = []
#END DEPENDENT VARIABLES


config ={

'SAMPLING_RATE': 15000,
'MIN_FREQ' : 220,
'MAX_RANGE' : 36,
'NUM_HARMONICS' : 20,

'TEMPO' : 80*np.random.rand() + 40,
'SMALLEST_NOTE' : 16,
'timeSig' : [4, 4],
'MEASURES' : 4,

'INSTRUMENT':'pure',

'RHYTHM_TEMPERATURE' : .5,
'PITCH_CHANGE_TEMPERATURE' : int(np.round(3*np.random.rand())) + 2, 

'keySig' : ['',''],
'CHORD_LIST' : {}, #chords that can be chosen from in this piece
'CHORD_SEQUENCE' : [],

'RHYTHM_BASED_WEIGHTS' : [],

'PLAYBACK_FREQS' : []
}

#initialize helpful variables dependent on config
def initDVs():
	global notesPerMeasure
	global notesPerPiece
	global DURATION
	global FREQS
	global FREQ_LETTERS
	global PROX_BASED_WEIGHTS
	notesPerMeasure = int(config['SMALLEST_NOTE']*config['timeSig'][0]/config['timeSig'][1])
	notesPerPiece = config['MEASURES']*notesPerMeasure
	DURATION = 60*config['timeSig'][1]/(config['SMALLEST_NOTE']*config['TEMPO'])

	minFreq = config['MIN_FREQ']
	FREQS = [(minFreq * np.power(2, x/12)) for x in range(config['MAX_RANGE'])]

	keys = list(keySigs.keys())
	values = list(keySigs.values())
	FREQ_LETTERS = [''+str(keys[values.index(i%12)])+str(int(((i+9)/12)+3)) for i in range(len(FREQS))]

	PCT = config['PITCH_CHANGE_TEMPERATURE']
	PROX_BASED_WEIGHTS = [(np.exp(-.5*np.square((x - 36)/PCT))) for x in range(config['MAX_RANGE']*2)]

def initMusicGenerator(key=['A','major'],randomKey=False,chordSequence=[]):
	initDVs()
	initInverseKeySigs()
	#initRhythmWeights(allEqual=True)
	initSpectraAndDecays()

	if len(config['keySig'][0]) < 1:
		initKeySig(key,randomKey)
	
	if len(config['CHORD_LIST'].keys()) < 1:
		initChordList()
		print(config['CHORD_LIST'])
		
	if len(config['CHORD_SEQUENCE']) == 0:
		if len(chordSequence) < 1:
			config['CHORD_SEQUENCE'] = createRandomChordSequence()
		else:
			config['CHORD_SEQUENCE'] = createSpecificChordSequence(chordSequence)

def initKeySig(key,random=False):
	keys = [key for key in keySigs.keys()]
	#global keySig
	if random:
		config['keySig'] = [np.random.choice(keys), np.random.choice(['major','minor'])]
	else:
		config['keySig'] = key
	print(config['keySig'])

#orinigally a one-time call, but now can be used to create chord-based weights
def initFreqWeights(oneOctaveWeights, key='A'):
	weights = []
	for i in range(4):
		weights.append(oneOctaveWeights)
	weights = np.array(weights).flatten().astype(np.float32)
	keyNum = keySigs[key]
	ans = weights[(12-keyNum):(12-keyNum + config['MAX_RANGE'])]
	ans /= ans.sum()
	return ans

#one-time call, NEEDS ATTENTION/HONING
# def initRhythmWeights(scalar=RHYTHM_TEMPERATURE,allEqual=False):
# 	global RHYTHM_BASED_WEIGHTS
# 	i = 0
# 	n = 1
# 	RHYTHM_BASED_WEIGHTS = np.ones(int(notesPerMeasure))
# 	if allEqual:
# 		return RHYTHM_BASED_WEIGHTS
# 	while i < notesPerMeasure:
# 		#print(i)
# 		#rhythmWeights[i] = (1 - np.power(scalar, n))/(1 - scalar)
# 		RHYTHM_BASED_WEIGHTS[i] = np.power(n,1)
# 		j = i
# 		while j > 0:
# 			RHYTHM_BASED_WEIGHTS[j] = RHYTHM_BASED_WEIGHTS[i]
# 			j -= 2*(notesPerMeasure - i)
# 		i = int(i + int((notesPerMeasure + 1- i))/2)
# 		n += 1
# 	#rhythmWeights = np.power(rhythmWeights, 2)
# 	#print(rhythmWeights)

#one-time call
def initSpectraAndDecays():
	global SPECTRA
	global DECAYS
	NUM_HARMONICS = config['NUM_HARMONICS']
	SPECTRA = {'pure': [1],
					'clarinet': [((1 + i)%2)/np.power(2,i/2) for i in range(NUM_HARMONICS)],
					'guitar': [7, 16, 14],
					'quadratic':[-np.square(i-NUM_HARMONICS/2) + np.square(NUM_HARMONICS/2)for i in range(NUM_HARMONICS)]}
	DECAYS = {'guitar': .0002}
	for s in SPECTRA:
		SPECTRA[s].append(np.zeros(NUM_HARMONICS - len(SPECTRA[s])))
		SPECTRA[s] = np.hstack(SPECTRA[s]).tolist()

#one-time call, essentially caching
def initInverseKeySigs():
	#global inverseKeySigs
	for key in keySigs:
		val = str(keySigs[key])
		if val not in inverseKeySigs.keys():
			inverseKeySigs[val] = [key]
		else:
			inverseKeySigs[val].append(key) 

#one-time call (or n for n number of key signatures in a piece)
def initChordList():
	CHORD_LIST = config['CHORD_LIST']
	keySig = config['keySig']
	chords = {}
	noteDict = CHORD_COMBOS[keySig[1]]
	keys = [key for key in noteDict.keys()]
	for note in keys:
		letter = inverseKeySigs[str((keySigs[keySig[0]] + int(note))%12)][0]
		for chord in noteDict[note]:
			CHORD_LIST[letter] = {}
			#print(letter)
			CHORD_LIST[letter][chord] = initFreqWeights(CHORDS[chord],letter).tolist()
	#print(config['CHORD_LIST'])

#called for one particular frequency
def makeWave(freq=440, numNotes=1, samplingRate=config['SAMPLING_RATE'], instrument=config['INSTRUMENT'], playChord=True, chords=[['A','M']]):
	n = DURATION*numNotes*samplingRate
	#print('\n\n',freq)
	harmonics = SPECTRA[instrument]
	data = np.zeros(int(n))
	for h in range(config['NUM_HARMONICS']):
		harm = harmonics[h]
		for i in range(len(data)):
			data[i] += harm * np.sin(2*(np.pi)*freq*(h+1)*i/samplingRate)
	if instrument in DECAYS:
		decay = DECAYS[instrument]
		for i in range(len(data)):
			data[i] *= np.exp(-1*i*decay)
	chordData = np.zeros(int(n))
	if (playChord):
		
		#print(chordSeq)
		for c in range(len(chords)):
			chord = chords[c]
			#print(chord)
			for i in range(int(DURATION*samplingRate)):
				offset = int(DURATION*samplingRate)*c
				for j in range(len(CHORDS[chord[1]])):
					chordfreq = FREQS[keySigs[chord[0]]+j]
					#print(chordfreq)
					chordData[i+offset] += CHORDS[chord[1]][j]*np.sin(2*(np.pi)*chordfreq*i/samplingRate)
		chordData /= np.amax(chordData)
	data += (.5*chordData)
	#plotWave(chordData)
	data /= np.amax(data)
	return data

def playFreqs(freqs=[], metronome=True,instrument='pure',repetitions=2):
	waves = []#np.array([0 for freq in freqs])
	i = 0
	while i < len(freqs):
		notes = 1
		chordsForNotes = [config['CHORD_SEQUENCE'][i]]
		for j in range(i + 1,len(freqs)):
			if freqs[j] == freqs[i]:
				notes = notes + 1
				chordsForNotes.append(config['CHORD_SEQUENCE'][j])
			else:
				break
		waves.append(makeWave(freq=freqs[i],numNotes=notes,instrument=instrument,chords=chordsForNotes))
		i += notes
	waves = np.array(waves)
	
	#flat = np.hstack(waves)
	#this accomplishes the same thing as the code below:
	flat = []
	for i in waves:
		for j in i:
			flat.append(j)
	flat = np.array(flat)
	#end

	if metronome:
		i = 0
		while i < len(flat):
			for j in range(i,i+25):
				if j < len(flat):
					flat[j] += 5
			i += int(config['SAMPLING_RATE']*DURATION*config['SMALLEST_NOTE']/config['timeSig'][1])
		flat /= np.amax(flat)

	PyAudio = pa.PyAudio
	p = PyAudio()
	stream = p.open(format = pa.paFloat32, 
	                channels = 1, 
	                rate = config['SAMPLING_RATE'], 
	                output = True)
	wave = flat.astype(np.float32).tobytes()# 	(data*2147483648).astype(np.float32).tobytes()

	for r in range(repetitions):
		chunk = 0
		CHUNK_SIZE = int(len(wave)/notesPerPiece)
		while chunk < len(wave) - 1:
			stream.write(wave[chunk:chunk+CHUNK_SIZE])
			chunk += CHUNK_SIZE
	stream.stop_stream()
	stream.close()
	p.terminate()
	plotWave(flat)
	return flat

#create freqs based on previous freq and overall
def createFreqs(keyBasedWeights,chordSequence=[],useChordSeq=True):
	MAX_RANGE = config['MAX_RANGE']
	CHORD_LIST = config['CHORD_LIST']

	frequencies = np.zeros(int(notesPerPiece))
	for i in range(int(notesPerPiece)):
		if i == 0:
			p = keyBasedWeights * config['CHORD_LIST'][chordSequence[i][0]][chordSequence[i][1]]
			p /= np.sum(p)
			frequencies[i] = np.random.choice(FREQS, size=1, p=p)
		else:
			prevFreq = int(np.round(12*np.log2(frequencies[i-1]/220)))#FREQS.index(frequencies[i-1])
			uniquePBW = PROX_BASED_WEIGHTS[(MAX_RANGE-prevFreq):(2*MAX_RANGE-prevFreq)]
			combinedProbs = np.multiply(keyBasedWeights, uniquePBW)
			#combinedProbs[np.argmax(combinedProbs)] *= RHYTHM_BASED_WEIGHTS[i%notesPerMeasure] #boosts the likelihood of playing same freq

			if useChordSeq:
				combinedProbs += np.array(config['CHORD_LIST'][chordSequence[i][0]][chordSequence[i][1]])*.1
			#print(combinedProbs)

			combinedProbs /= combinedProbs.sum()
			#printProbs(i,combinedProbs)
			frequencies[i] = np.random.choice(FREQS, size=1, p=combinedProbs)
	return frequencies

def printProbs(note,combinedProbs):
	print("Note "+str(note)+":")
	for i in range(len(combinedProbs)):
		print(FREQ_LETTERS[i]+': '+str(combinedProbs[i]))

#return 1d array of named chords
def createRandomChordSequence():
	chordSeq = []
	i = 0
	letters = [key for key in config['CHORD_LIST'].keys()]
	while i < notesPerPiece:
		n = np.random.choice([.5,1,2])
		length = int(n*notesPerMeasure)
		if length > notesPerPiece - i:
			length = notesPerPiece - i
		letter = np.random.choice(letters)
		possibleChords = [key for key in config['CHORD_LIST'][letter].keys()]
		chord = [letter,np.random.choice(possibleChords)]
		for x in range(i, i + length):
			chordSeq.append(chord)
		i += length
	#print(len(chordSeq))
	return chordSeq

#takes an array spreadOutChords and spreads them out evenly over notesPerPiece
def createSpecificChordSequence(spreadOutChords):
	note = 0
	answer = []
	length = int(notesPerPiece/len(spreadOutChords))
	print(length)
	while note < notesPerPiece:
		index = int(note/length)
		print(index)
		answer.append(spreadOutChords[index])
		note = note + 1
	print(answer)
	return answer

def plotWave(wavedata):
	portion = 1
	x = np.linspace(0,len(wavedata)/config['SAMPLING_RATE']*portion,num=len(wavedata)*portion)
	y = wavedata[0:int(len(wavedata)*portion)]
	plt.plot(x,y)
	plt.show()

def writeFile(filename,function='a'):
	with open(filename,function) as file:
		json.dump(config, file)

	#file.write(str([FREQ_LETTERS[FREQS.index(freq)] for freq in freqs])+'\n\n')

#START OF MAIN
#config is already initialized with default values

if len(sys.argv) > 1:
	data = json.load(open(sys.argv[1], encoding = 'utf8'))
	for key in data:
		config[key] = data[key]

initMusicGenerator(randomKey=True)#key=['E','major'],chordSequence=[['B','M'],['A','M'],['B','M'],['A','M']])

#if playback data already exists, just play it
if len(config['PLAYBACK_FREQS']) > 0:
	print(config)
	playFreqs(config['PLAYBACK_FREQS'])
	filename = 'data/randomMusic.txt'
	file = writeFile(filename,'w')
else:
	keyBasedWeights = initFreqWeights(SCALES[config['keySig'][1]], key=config['keySig'][0])
	config['PLAYBACK_FREQS'] = createFreqs(keyBasedWeights,config['CHORD_SEQUENCE']).tolist()
	wave = playFreqs(config['PLAYBACK_FREQS'], instrument=config['INSTRUMENT'], repetitions=2).tolist()
	#plotWave(wave)
	filename = 'data/randomMusic.txt'
	file = writeFile(filename,'w')


END_TIME = time.time()
print(END_TIME - START_TIME, 'seconds')