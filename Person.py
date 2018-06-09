import numpy as np

class Person(object):
    tracks = []
    def __init__(self, tl, br, id):
        self.tl = tl
        self.br = br
        self.id = id

    def getTracks(self):
        return self.tracks

    def getID(self):
        return self.id

    def updateCoords(self, tl, br):
        self.tracks.append([self.tl, self.br])
        self.tl = tl
        self.br = br






