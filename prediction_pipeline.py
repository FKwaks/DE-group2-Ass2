from __future__ import absolute_import
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import argparse
import csv
import io
import logging
import math
import json

import apache_beam as beam
import pandas as pd
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def get_csv_reader(readable_file):
    field_list = ['listing_id','id','date','reviewer_id','reviewer_name','comments']

    # Open a channel to read the file from GCS
    gcs_file = beam.io.filesystems.FileSystems.open(readable_file)
    
    # Return the csv reader
    return csv.DictReader(io.TextIOWrapper(gcs_file, encoding = 'utf-8'), field_list)

class MyPredictDoFn(beam.DoFn):

    def __init__(self):
        self._model = SentimentIntensityAnalyzer()
        self.entry = None
        
    def calculate_sentiment(self, entry):
        sid_obj = SentimentIntensityAnalyzer() 
        if (type(self.entry) != str and math.isnan(entry)):
            return -55
        opinion = sid_obj.polarity_scores(self.entry)
        return opinion['compound']

    def process(self, elements, **kwargs):

        df = pd.DataFrame(elements)
        df = df.iloc[1:]
        print(df)
        new_list = []
        sid_obj = SentimentIntensityAnalyzer()
        
        for i in list(df.comments):
            if (type(i) != str and math.isnan(i)):
                i = int(-55)    
            opinion = sid_obj.polarity_scores(i)
            new_list.append(opinion['compound'])

        df['comments_'] = new_list
        df = df[df['comments_'] != -55]
        df = df.groupby('date')['comments_'].mean()
        print(df)
        df = pd.DataFrame({'date':df.index, 'value':df.values})
        print('Dit zijn de kolommen in het dataframe {}'.format(df.columns))
        print(df)

        result = df.to_json(orient="records")
        parsed = json.loads(result)
        return json.dumps(parsed, indent=4)


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""
    parser = argparse.ArgumentParser()

    known_args, pipeline_args = parser.parse_known_args(argv)

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(options=pipeline_options) as p:
        
        # Read the text file[pattern] into a PCollection.
        prediction_data = (p | 'CreatePCollection' >> beam.Create([known_args.input])
                           | 'ReadCSVFile' >> beam.FlatMap(get_csv_reader))

        # https://beam.apache.org/releases/pydoc/2.25.0/apache_beam.transforms.util.html#apache_beam.transforms.util.BatchElements
        # https://beam.apache.org/documentation/transforms/python/aggregation/groupintobatches/
        output = (prediction_data
                  | 'batch into n batches' >> beam.BatchElements(min_batch_size=1000, max_batch_size=1001)
                  | 'Predict' >> beam.ParDo(MyPredictDoFn()))

        output | 'WritePredictionResults' >> WriteToText(known_args.output, file_name_suffix=".json")

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
