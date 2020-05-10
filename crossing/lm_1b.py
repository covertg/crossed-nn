# NOTICE that some of this code is derived from the work of the Tensorflow Authors under Apache 2.0 (2016).
# Original work: https://github.com/tensorflow/models/blob/master/research/lm_1b
# The vocabulary and model loading have been updated for Tensorflow v2.
# The softmax functions are original.

from crossing import data
from google.protobuf import text_format
import numpy as np
import tensorflow as tf

MAX_WORD_LEN = 50  # Number of characters. Fixed value from Google's demo
PUNCT = data.punct + ('.',)


def _preprocess_sentence(sent, idx):
    idx_pad = 0
    # add bos and eos tokens
    if sent.find('<S>') != 0:
        sent = '<S> ' + sent
        idx_pad += 4
    if '</S>' not in sent:
        sent = sent + ' </S>'
    # ensure spacing between punctuation
    idx_pad += sum([sent[idx:].count(p) for p in PUNCT])
    split = [(w[:-1] + ' ' + w[-1] + ' ') if w.endswith(PUNCT) else (w + ' ') for w in sent.split()]
    return ''.join(split), idx+idx_pad  # This results in an extra space after the sentence, but that's fine / the vocab trims it


def get_model_fn(pbtxt, ckpt, vocab_file):
    vocab = CharsVocabulary(vocab_file, MAX_WORD_LEN)
    sess, t = _LoadModel(pbtxt, ckpt)
    print('Loaded model and vocab for lm_1b.')

    def model_fn(sent, idx):
        sent, idx = _preprocess_sentence(sent, idx)
        # Tokenize sentence
        sent_id = [vocab.word_to_id(w) for w in sent.split()]  # List of integers
        chars_id = [vocab.word_to_char_ids(w) for w in sent.split()]  # List of numpy arrays of shape (MAX_WORD_LEN,) = (50,)
        surprisal_idx = len([vocab.word_to_id(w) for w in sent[:idx].split()])
        # Compute conditional probability over each increasing sequence of words
        probs = softmax_probs(sess, t, sent_id, chars_id)
        log_probs = tf.nn.log_softmax(probs, axis=1)
        # Total surprisal over region
        surprisals = []
        for i, tok in enumerate(sent_id):
            surprisals.append(-1 * log_probs[i][tok])
        surprisals = surprisals[surprisal_idx:]  # Region of interest is from surprisal_idx till the end
        total = tf.reduce_sum(surprisals).numpy()
#         print('[lm_1b.py] {:.2f} \t{}'.format(total, sent))
        return total

    return model_fn


def softmax_probs(sess, t, sent_id, chars_id):
    outputs = []
    sess.run(t['states_init'])
    for word, chars in zip(sent_id, chars_id):
        softmax = sess.run(t['softmax_out'], feed_dict={
            # Shape (1, 1, MAX_WORD_LEN) = (1, 1, 50)
            t['char_inputs_in']: np.expand_dims(chars, (0, 1)),
            t['inputs_in']: np.array([[word]]),  # Shape (1, 1)
            t['targets_in']: np.zeros([1, 1], np.int32),
            t['target_weights_in']: np.ones([1, 1], np.float32)
        })  # Numpy array of shape (1, 800k), type float32
        outputs.append(softmax)
    return tf.concat(outputs, axis=0)  # Shape is (len(sent_id), 800k)


def _LoadModel(gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.

    Args:
      gd_file: GraphDef proto text file.
      ckpt_file: TensorFlow Checkpoint file.

    Returns:
      TensorFlow session and tensors dict.
    """
    with tf.Graph().as_default():
        # Create GraphDef object from gd_file
        print('Reading graph %s' % gd_file)
        with tf.io.gfile.GFile(gd_file, 'r') as f:
            s = f.read()
        gd = tf.core.framework.graph_pb2.GraphDef()  # Previously tf.GraphDef()
        text_format.Merge(s, gd)

        # Populate dictionary of Tensors/Operations from GraphDef
        print('Creating tensors.')
        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
         ] = tf.import_graph_def(gd, {}, [
             'states_init',
             'lstm/lstm_0/control_dependency:0',
             'lstm/lstm_1/control_dependency:0',
             'softmax_out:0',
             'class_ids_out:0',
             'class_weights_out:0',
             'log_perplexity_out:0',
             'inputs_in:0',
             'targets_in:0',
             'target_weights_in:0',
             'char_inputs_in:0',
             'all_embs_out:0',
             'Reshape_3:0',
             'global_step:0'],
            name='')

        print('Loading, initializing weights %s' % ckpt_file)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True))  # Place ops on CPU if no GPU is present
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return sess, t


class Vocabulary(object):
    """Class that holds a vocabulary for the dataset."""

    def __init__(self, filename):
        """Initialize vocabulary.

        Args:
          filename: Vocabulary file name.
        """

        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        print('Reading vocabulary %s' % filename)
        with tf.io.gfile.GFile(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        if cur_id < self.size:
            return self._id_to_word[cur_id]
        return 'ERROR'

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence):
        """Convert a sentence to a list of ids, with special tokens added."""
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class CharsVocabulary(Vocabulary):
    """Vocabulary containing character-level information."""

    def __init__(self, filename, max_word_length):
        super(CharsVocabulary, self).__init__(filename)
        self._max_word_length = max_word_length
        chars_set = set()

        for word in self._id_to_word:
            chars_set |= set(word)

        free_ids = []
        for i in range(256):
            if chr(i) in chars_set:
                continue
            free_ids.append(chr(i))

        if len(free_ids) < 5:
            raise ValueError('Not enough free char ids: %d' % len(free_ids))

        self.bos_char = free_ids[0]  # <begin sentence>
        self.eos_char = free_ids[1]  # <end sentence>
        self.bow_char = free_ids[2]  # <begin word>
        self.eow_char = free_ids[3]  # <end word>
        self.pad_char = free_ids[4]  # <padding>

        chars_set |= {self.bos_char, self.eos_char, self.bow_char, self.eow_char,
                      self.pad_char}

        self._char_set = chars_set
        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros(
            [num_words, max_word_length], dtype=np.int32)

        self.bos_chars = self._convert_word_to_char_ids(self.bos_char)
        self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = ord(self.pad_char)

        if len(word) > self.max_word_length - 2:
            word = word[:self.max_word_length-2]
        cur_word = self.bow_char + word + self.eow_char
        for j in range(len(cur_word)):
            code[j] = ord(cur_word[j])
        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence):
        chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


def get_batch(generator, batch_size, num_steps, max_word_length, pad=False):
    """Read batches of input."""
    cur_stream = [None] * batch_size

    inputs = np.zeros([batch_size, num_steps], np.int32)
    char_inputs = np.zeros([batch_size, num_steps, max_word_length], np.int32)
    global_word_ids = np.zeros([batch_size, num_steps], np.int32)
    targets = np.zeros([batch_size, num_steps], np.int32)
    weights = np.ones([batch_size, num_steps], np.float32)

    no_more_data = False
    while True:
        inputs[:] = 0
        char_inputs[:] = 0
        global_word_ids[:] = 0
        targets[:] = 0
        weights[:] = 0.0

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        cur_stream[i] = list(generator.next())
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
                global_word_ids[i,
                                cur_pos:next_pos] = cur_stream[i][2][:how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]
                weights[i, cur_pos:next_pos] = 1.0

                cur_pos = next_pos
                cur_stream[i][0] = cur_stream[i][0][how_many:]
                cur_stream[i][1] = cur_stream[i][1][how_many:]
                cur_stream[i][2] = cur_stream[i][2][how_many:]

                if pad:
                    break

        if no_more_data and np.sum(weights) == 0:
            # There is no more data and this is an empty batch. Done!
            break
        yield inputs, char_inputs, global_word_ids, targets, weights
