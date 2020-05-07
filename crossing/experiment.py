from crossing import data
import numpy as np
import pandas as pd

output_cols = ['Surprisal', 'Preposition', 'E/D', 'Connective', 'Crossing']

def _get_expansion_choices(sent):
    prep, ed, comp, conn = None, None, None, None
    for s in data.expanders['{Prep}']:
        if s in sent:
            prep = s
            break
    for s in data.expanders['{E/D}']:
        if s in sent:
            ed = s
            break
    for s in data.expanders['{Comp}']:
        if s in sent:
            comp = s
            break
    for s in data.expanders['{Conn}']:
        if s in sent:
            conn = s
            break
    return prep, ed, comp, conn

def _get_surprisal_idx(sent):
    split = sent.split(';') if ';' in sent else sent.split('.')
    i1 = len(split[0])
    i2 = 1 + split[1].index(',')
    return i1 + i2


def run_experiment(inputs, model_fn, outfile):
    outputs = []
    for i, line in inputs.iterrows():
        for c in data.cols[1:]:
            sents = line[c]
            if sents:
                sents = sents.split(data.DELIMITER)
                for sent in sents:
                    # cumulative surprisal from model
                    surprisal = model_fn(sent, _get_surprisal_idx(sent))
                    # get expansion choice. this is messy, but I realized it would be nice to have this pretty late.
                    prep, ed, _, connective = _get_expansion_choices(sent)
                    crossing = c
                    outputs.append([surprisal, prep, ed, connective, crossing])
                    # print('New line is ', outputs[-1])
    # And when we're all done, write out to a df
    df = pd.DataFrame.from_records(outputs, columns=output_cols)
    df.to_csv(outfile, sep='\t', index=False)