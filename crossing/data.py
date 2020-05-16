import pandas as pd

cols = ['Base', 'E_nX', 'E_X', 'C_nX', 'C_X']
punct = (';', ',') 
conn_summ, conn_contr = ['in short', 'therefore', 'that is', 'thus'], ['on the other hand', 'by contrast']
connectives = conn_summ + conn_contr
expanders = {
    '{Prep}': ['Near', 'By', 'Nearby'],
    '{E/D}': ['Here is', 'This is', 'That is'],  # ['Here is', 'There is', 'That is', 'This is', 'It is']
    '{Comp}': ['that'],  # Do we want "which?" I don't think so.
    '{Conn}': [p + ' ' + c + ', ' for c in connectives for p in punct]
}


# Returns a list of dicts; each tuple becomes a dict
def read_items(filename):
    df = pd.read_csv(filename, sep='\t')
    assert ~df.isna().values.any(), 'Input CSV has NA'
    return df  # Columns are (A, B, Vtr, Vin1, Vin2)
    # return df.to_dict('records')


# Returns a list of dataframes
def read_templates(*filenames, delimeter='|'):
    templates = []
    for f in filenames:
        # Read into pandas df
        df = pd.read_csv(f, sep='\t', na_filter=False)
        # Expand the last four columns into lists of strings, rather than single strings
        for c in cols:
            df[c] = df[c].apply(lambda s: []) if (not df[c].any()) else df[c].apply(lambda s: s.split(delimeter))
        templates.append(df)
    return templates


def read_sents(filename):
    return pd.read_csv(filename, sep='\t', na_filter=False)
    

# Joins a string list and adds a space or | character in-between strings
def _join(str_list, delimiter=' '):
    return str_list[0] + ''.join([delimiter + w for w in str_list[1:]]) if str_list else None


def _map(item, l):
    return _join([item[w] if (w in item.keys()) else w for w in l.split(' ')])


def expand_template(items, template, capitalize=True):
    # First we expand on the template by creating all inferences (premise-entailment pairs) that it affords
    sents_intermediate = []
    for i, item in items.iterrows():
        for _, line in template.iterrows():
            base_premise = _map(item, line[0][0])
            base_premise = base_premise[0].upper() + base_premise[1:] if capitalize else base_premise
            for label, entailments in line[1:].iteritems():
                correct = 'E' in label
                crossing = 'n' in label
                for entailment in entailments:
                    subform = int(entailment[0])
                    entailment = _map(item, entailment[2:])
                    sentence = base_premise + '{Conn}' + entailment + '.'   # Whole sentence aka "inference"
                    sents_intermediate.append([sentence, i, correct, crossing, subform])    
    data_cols = ['inference', 'item', 'correct', 'crossing', 'subform.id']
    df_intermediate = pd.DataFrame.from_records(sents_intermediate, columns=data_cols)

    # Then we expand on {curly brace} expanders, such as connectives.
    sents_expanded = []
    for intermed in sents_intermediate:
        expandeds = _expand(intermed[0])
        new_sents = [(expanded[0], *intermed[1:], *expanded[1].values()) for expanded in expandeds]
        sents_expanded.extend(new_sents)
        if 'surprisal.idx' not in data_cols:
            data_cols.extend(expandeds[0][1].keys())
    df_expanded = pd.DataFrame.from_records(sents_expanded, columns=data_cols)

    return df_intermediate, df_expanded


# Expands on {expanders}, returning a list of (sentence, attribute) pairs, where 'attribute' is the choice(s) of expander made.
def _expand(sentence):
    stack = [(sentence, dict.fromkeys(expanders.keys()))]
    expanded = []
    while stack:
        sent, attributes = stack.pop()
        if '{' in sent:
            for k in expanders.keys():
                if k in sent:
                    for value in expanders[k]:
                        new_sent = sent[:sent.index(k)] + value + sent[sent.index(k)+len(k):]
                        attributes.update({k: value})
                        if k == '{Conn}':
                            attributes.update({'connective.summ': (value[2:-2] in conn_summ)})  # 2:-2 strips punctuation
                            attributes.update({'surprisal.idx': len(sent[:sent.index(k)] + value)})  # experiment.py assumes that surprisal.idx is the last column
                        stack.append((new_sent, attributes.copy()))
        else:
            expanded.append((sent, attributes))
    return expanded