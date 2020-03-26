import pandas as pd
import urllib.request

def url_to_jpg(i, url, file_path):
    fileName='image-{}.jpg'.format(i)
    full_path-'{}{}'.format(file_path,fileName)
    urllib.request.urlretrieve(url, full_path)

    print('{} saved.'.format(fileName))
    return none

FILENAME = 'result.csv'

FILE_PATH = 'New folder/'

urls = pd.read_csv(FILENAME)

for i,url in enumerate(urls.values):
    url_to_jpg(i, url[0], FILE_PATH)