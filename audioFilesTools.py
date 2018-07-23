# -*- coding: utf-8 -*-
import eyed3
import csv
#Remove logs
eyed3.log.setLevel("ERROR")

songDict = {}
with open('songMetaData.csv','r') as csvfile:
	reader = csv.reader(csvfile, delimiter = ",")
	for row in reader:
		name = row[1].strip()
		songDict[name] = [row[0],name,row[2]]


def isMono(filename):
	audiofile = eyed3.load(filename)
	return audiofile.info.mode == 'Mono'

def getGenre(filename):
	try:
		return songDict[filename][2]
	except:
		return "Unknown"



	