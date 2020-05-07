import pandas as pd

DELIMITER = '|'

cols = ['Base', 'E_nX', 'E_X', 'C_nX', 'C_X']

connectives = ['I repeat', 'again', 'in short', 'therefore', 'that is', 'thus']
expanders = {
    '{Prep}': ['Near', 'By', 'Nearby'],  # ['near', 'nearby', 'by']
    '{E/D}': ['Here is', 'This is'],  # ['Here is', 'There is', 'That is', 'This is', 'It is']
    '{Comp}': ['that'],  # ['that', 'which']
    '{Conn}': ['; ' + c + ', ' for c in connectives] + ['. ' + c[0].upper() + c[1:] + ', ' for c in connectives]
}

# Returns a list of dicts; each tuple becomes a dict
def read_items(filename):
    df = pd.read_csv(filename, sep='\t')
    assert ~df.isna().values.any(), 'Input CSV has NA'
    return df.to_dict('records')  # Each dict has keys A, B, Vtr, Vin1, Vin2


# Returns a list of dataframes
def read_templates(*filenames):
    templates = []
    for f in filenames:
        # Read into pandas df
        df = pd.read_csv(f, sep='\t', na_filter=False)
        # Expand the last four columns into lists of strings, rather than single strings
        for c in cols:
            df[c] = df[c].apply(lambda s: []) if (not df[c].any()) else df[c].apply(lambda s: s.split(DELIMITER))
        templates.append(df)
    return templates


def _map(item, w):
    return item[w] if (w in item.keys()) else w


# Returns a dataframe much like template, but phrases are split into string-arrays instead of full strings
def expand_template(items, template):
    """
    For each tuple, create all possible phrases defined by each column of the template.
    """
    expanded_lines = []
    for item in items:  # Recall that item is a dict, and will serve as a map from templateland to languageland
        for i, line in template.iterrows():
            phrases_per_line = []
            for c in cols:
                # "Forms" may be a Base (1st column) or an implication (rest of the columns).
                # After base, each column may be empty, contain one form, or contain multiple form. Hence line[c] is a list.
                forms = line[c] 
                if forms:
                    forms = [s.split(' ') for s in forms]
                    forms = [[_map(item, w) for w in s] for s in forms]
                phrases_per_line.append(forms)
            # Now phrases_per_line contains all the phrases given by each column, for one line (row) of the template
            # base, e_nX, e_X, c_nX, c_X = phrases_per_line  # e.g. if we were to unpack this line
            expanded_lines.append(phrases_per_line)
    # Note that one "item" or original line from the tuples yields many sentences/expanded lines
    return pd.DataFrame.from_records(expanded_lines, columns=cols)


# Joins a string list and adds a space or | character in-between strings
def _join(str_list, delimiter=' '):
    return str_list[0] + ''.join([delimiter + w for w in str_list[1:]]) if str_list else None


def form_unexpanded_sentences(items, template, capitalize=True):
    phrases = expand_template(items, template)
    unexpanded_sentences = []
    for i, line in phrases.iterrows():
        base = _join(line[cols[0]][0])
        base = (base[0].upper() + base[1:]) if capitalize else base
        sentences_per_line = []
        for c in cols[1:]:
            implications = line[c]
            if implications:
                implications = [_join(s) for s in implications]
            sentences = [(base + ' {Conn} ' + s + '.') for s in implications]
            sentences = _join(sentences, delimiter=DELIMITER)  # Use a special delimiter, since Pandas will serialize lists/strings on writing to file
            sentences_per_line.append(sentences)
            # print('adding {} for {}'.format(sentences, c))
        unexpanded_sentences.append(sentences_per_line)
    return pd.DataFrame.from_records(unexpanded_sentences, columns=cols[1:])


def _expand(sentences):
    if sentences:
        unexpanded = sentences.split(DELIMITER)
        expanded = []
        stack = unexpanded
        while stack:
            sent = stack.pop()
            if '{' in sent:
                for e in expanders.keys():
                    if e in sent:
                        expansions = [sent[:sent.index(e)] + value + sent[sent.index(e)+len(e):] for value in expanders[e]]
                        stack.extend(expansions)
            else:
                expanded.append(sent)
        return _join(expanded, delimiter=DELIMITER)
    return None


# Our data's *final form* is realized by expanding on "expanders" in curly braces and finally putting one sentence for line
# (The one sentence per line thing ended up being mostly for programming ease. It would be nicer for convenient statistical analysis to group the sentences together better.)
def expand_sentences(unexpanded_sentences):
    # We have a line
    # We go through each column
    # We split the big sent by |, yielding either one or two sentences (if nonempty)
    # We resolve curly {}s by creating new sentences, to be kept on the same line, |-delimited
    # Ultimate dataframe has header E_nX E_X C_nX C_X, with likely multiple sentences in each column
    
    # ---- actually we can do this with apply() much more easily
    df = unexpanded_sentences.copy(deep=True)
    for c in cols[1:]:
        df[c] = df[c].apply(_expand)
    return df


def read_sents(filename):
    return pd.read_csv(filename, sep='\t', na_filter=False)