# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond

# Tweakable parameters
desiredSize = 128

# Define
currentPath = os.path.dirname(os.path.realpath(__file__))

# Remove logs
eyed3.log.setLevel("ERROR")

# Create spectrogram from mp3 files


def createSpectrogram(filename, newFilename):
    # Create temporary mono track if needed
    ismon = isMono(rawDataPath + filename)
    if ismon == 'BAD':
        print ('corrupt file!')
        return None
    if ismon:
        command = "cp '{}' 'tmp/{}.mp3'".format(
            rawDataPath + filename, newFilename)
    else:
        command = '"sox" "{}" "tmp/{}.mp3" remix 1,2'.format(
            rawDataPath + filename, newFilename)
    print (command)
    p = Popen(command, shell=True, stdin=PIPE,
              stdout=PIPE, stderr=STDOUT, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        print (errors)

    # Create spectrogram
    filename.replace(".mp3", "")
    command = '"sox" "tmp/{}.mp3" -n spectrogram -Y 200 -X {} -m -r -o "{}.png"'.format(
        newFilename, pixelPerSecond, spectrogramsPath + newFilename)
    p = Popen(command, shell=True, stdin=PIPE,
              stdout=PIPE, stderr=STDOUT, cwd=currentPath)
    print(command)
    output, errors = p.communicate()
    if errors:
        print (errors)

    try:
        # Remove tmp mono track
        os.remove("tmp/{}.mp3".format(newFilename))
    except Exception as e:
        print (str(e))

# Creates .png whole spectrograms from mp3 files


def createSpectrogramsFromAudio():
    genresID = dict()
    files = os.listdir(rawDataPath)
    files = [file for file in files if file.endswith(".mp3")]
    nbFiles = len(files)

    # Create path if not existing
    if not os.path.exists(os.path.dirname(spectrogramsPath)):
        try:
            os.makedirs(os.path.dirname(spectrogramsPath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Rename files according to genre
    for index, filename in enumerate(sorted(files)):
        print ("Creating spectrogram for file {}/{}...".format(index + 1, nbFiles))
        if len(filename) > 12 and filename[12] == '~':
            filename_to_genre = filename[13:]
        else:
            filename_to_genre = filename
        fileGenre = getGenre(filename_to_genre)
        if fileGenre == 'Unknown':
            continue
        genresID[fileGenre] = genresID[fileGenre] + \
            1 if fileGenre in genresID else 1
        fileID = genresID[fileGenre]
        newFilename = fileGenre + "_" + str(fileID)
        if not os.path.exists('Data/Spectrograms/{}.png'.format(newFilename)):
            createSpectrogram(filename, newFilename)

# Whole pipeline .mp3 -> .png slices


def createSlicesFromAudio():
    print ("Creating spectrograms...")
    createSpectrogramsFromAudio()
    print ("Spectrograms created!")

    print ("Creating slices...")
    createSlicesFromSpectrograms(desiredSize)
    print ("Slices created!")
