import sys
import urllib
from csv import reader
import os.path

csv_filename = sys.argv[1]

with open(csv_filename+".csv".format(csv_filename), 'r') as csv_file:
    for line in reader(csv_file):
        if os.path.isfile("fullres/" + line[0] + ".jpg"):
            print "Image skipped for {0}".format(line[0])
        else:
            if line[2] != '' and line[0] != "ImageID":
                urllib.urlretrieve(line[2], "fullres/" + line[0] + ".jpg")
                print "Image saved for {0}".format(line[0])
            else:
                print "No result for {0}".format(line[0])