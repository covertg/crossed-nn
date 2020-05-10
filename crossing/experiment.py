from crossing import data
import datetime
import numpy as np
import pandas as pd

def run_experiment(inputs, outfile, model_fn):
    outputs = []
    for _, line in inputs.iterrows():
        # Total surprisal from model
        surprisal = model_fn(line[0], line[-1])  # 0: sentence; -1: surprisal index
        outputs.append([surprisal, *line[1:]])
    # And when we're all done, write out to a df
    df = pd.DataFrame.from_records(outputs, columns=inputs.columns)
    df.to_csv(outfile, sep='\t', index=False)


def run_experiments(infilef, outfilef, model_fn, n):
    for i in range(n):
        print('[experiment.py] Experiment {}\t [START] {}'.format(i+1, datetime.datetime.now().time()), end='')
        inputs = data.read_sents(infilef % (i+1))
        run_experiment(inputs, outfilef % (i+1), model_fn)
        print('\t [END] {}'.format(datetime.datetime.now().time()))