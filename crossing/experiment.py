from crossing import data
import datetime
import numpy as np
import pandas as pd

def run_experiment(inputs, outfile, model_fn, silent=True):
    outputs = []
    for i, line in inputs.iterrows():
        # Total surprisal from model
        surprisal = model_fn(line[0], line[-1])  # 0: sentence; -1: surprisal index
        outputs.append([surprisal, *line[1:]])
        # Log every 200
        if not silent and not ((i+1) % 100):
            print('[experiment.py] Processed {} lines \t {}'.format(i+1, datetime.datetime.now().time()))
    # And when we're all done, write out to a df
    df = pd.DataFrame.from_records(outputs, columns=inputs.columns)
    df.to_csv(outfile, sep='\t', index=False)


def run_experiments(infilef, outfilef, model_fn, n_exps, silent=True):
    for i in n_exps:
        print('[experiment.py] Experiment {}\t [START] {}'.format(i+1, datetime.datetime.now().time()))
        inputs = data.read_sents(infilef % (i+1))
        run_experiment(inputs, outfilef % (i+1), model_fn, silent=silent)
        print('[experiment.py] Experiment {}\t [END] {}'.format(i+1, datetime.datetime.now().time()))