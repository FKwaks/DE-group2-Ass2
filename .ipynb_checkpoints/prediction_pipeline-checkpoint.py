from __future__ import absolute_import

import argparse
import csv
import io
import json
import logging

import apache_beam as beam
import pandas as pd
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from google.cloud import storage
from keras.layers import Dense
from keras.models import Sequential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 


def train_save_model(readable_file, project_id, bucket_name):
    # Open a channel to read the file from GCS
    gcs_file = beam.io.filesystems.FileSystems.open(readable_file)

    # Read it as csv, you can also use csv.reader
    csv_dict = csv.DictReader(io.TextIOWrapper(gcs_file))

    # Create the DataFrame
    df = pd.DataFrame(csv_dict)
    df = df.apply(pd.to_numeric)
    dataset = df.values
    
    sid_obj = SentimentIntensityAnalyzer() 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    
    
    text_out = {
        "Sentimental Compound value:": sentiment_dict['compound'],
    }

    return json.dumps(str(text_out), sort_keys=False, indent=4)


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        default='gs://dataflow-samples/Data/reviews.csv',
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        # CHANGE 1/6: The Google Cloud Storage path is required
        # for outputting the results.
        default='gs://YOUR_OUTPUT_BUCKET/AND_OUTPUT_PREFIX',
        help='Output file to write results to.')

    parser.add_argument(
        '--pid',
        dest='pid',
        help='project id')

    parser.add_argument(
        '--mbucket',
        dest='mbucket',
        help='model bucket name')
    known_args, pipeline_args = parser.parse_known_args(argv)
    print(known_args)
    print(pipeline_args)
    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(options=pipeline_options) as p:
        output = (p | 'Create FileName Object' >> beam.Create([known_args.input])
                  | 'TrainAndSaveModel' >> beam.FlatMap(train_save_model, known_args.pid, known_args.mbucket)
                  )
        output | 'Write' >> WriteToText(known_args.output)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()