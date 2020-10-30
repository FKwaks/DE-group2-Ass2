import urllib.request
import json

url = "https://storage.googleapis.com/de2020ass2grp2/results-00000-of-00001.json"
response = urllib.request.urlopen(url)
data = json.loads(response.read())
print (data)