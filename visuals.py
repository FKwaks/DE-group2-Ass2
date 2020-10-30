import urllib.request
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://storage.googleapis.com/de2020ass2grp2/results-00000-of-00001.txt"

def draw_plot(url):
    date = []
    value = []


    file = urllib.request.urlopen(url)
    for line in file:
        decoded_line = line.decode("utf-8")
        decoded_line = decoded_line[1:-2]
        words = decoded_line.split(', ')
        date.append(words[0])
        value.append(words[1])
    df = pd.DataFrame(data=list(zip(date,value)), columns=['date','value'])
    df = df.iloc[pd.to_datetime(df.date).values.argsort()]
    df = df.reset_index(drop=True)
    df.value = df.value.astype(float)
    print(df)

    plotje = sns.lineplot(x="date", y="value", data=df)
    plt.xticks(rotation=90)
    for ind, label in enumerate(plotje.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.title('sentiment analysis over time AirBNB 2019')
    plt.show()


draw_plot(url)


