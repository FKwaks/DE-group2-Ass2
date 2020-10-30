import urllib.request, json 

with urllib.request.urlopen("https://storage.googleapis.com/de2020ass2grp2/results-00000-of-00001.json") as url:
    data = json.loads(url.read().decode())
    print(data)