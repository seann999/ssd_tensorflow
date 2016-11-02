import cv2
#import urllib.request
import urllib2
import numpy as np
import threading
import time

class WebcamStream:
    def __init__(self, address):
        self.image = None
        self.address = address

    def start_stream(self):
        bytes = None

        print("starting stream...")
        stream = urllib2.urlopen(self.address) #'http://192.168.100.102:8080/video'
        bytes = b''

        while True:
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                self.image = cv2.cvtColor(cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                #cv2.imshow('i', self.image)
                #cv2.waitKey(1)

    def start_stream_threads(self):
        threading._start_new_thread(self.start_stream, ())
