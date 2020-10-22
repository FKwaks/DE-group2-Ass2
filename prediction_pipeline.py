#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import

import argparse
import csv
import io
import logging

import apache_beam as beam
import pandas as pd
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from google.cloud import storage
from keras.models import load_model


def get_csv_reader(readable_file):
    # Open a channel to read the file from GCS
    gcs_file = beam.io.filesystems.FileSystems.open(readable_file)

    # Return the csv reader
    return csv.DictReader(io.TextIOWrapper(gcs_file))


class MyPredictDoFn(beam.DoFn):

    def __init__(self, project_id, bucket_name):
        self._model = None
        self._project_id = project_id
        self._bucket_name = bucket_name

    def setup(self):
        logging.info("MyPredictDoFn initialisation. Load Model")
        client = storage.Client(project=self._project_id)
        bucket = client.get_bucket(self._bucket_name)
        blob = bucket.blob('models/model.h5')
        blob.download_to_filename('downloaded_model.h5')
        self._model = load_model('downloaded_model.h5')

    def process(self, element, **kwargs):
        df = pd.DataFrame.from_dict(element,
                                    orient="index").transpose().fillna(0)
        df = df.apply(pd.to_numeric)
        val_set2 = df.copy()
        logging.info(val_set2)
        result = self._model.predict(df)
        y_classes = result.argmax(axis=-1)
        val_set2['class'] = y_classes.tolist()
        dic = val_set2.to_dict(orient='records')
        return [dic]


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        default='gs://dataflow-samples/data/kinglear.txt',
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

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(options=pipeline_options) as p:
        # Read the text file[pattern] into a PCollection.
        prediction_data = (p | 'CreatePCollection' >> beam.Create([known_args.input])
                           | 'ReadCSVFle' >> beam.FlatMap(get_csv_reader))
        output = (prediction_data | 'Predict' >> beam.ParDo(MyPredictDoFn(project_id=known_args.pid,
                                                                          bucket_name=known_args.mbucket)))
        output | 'WritePR' >> WriteToText(known_args.output)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
