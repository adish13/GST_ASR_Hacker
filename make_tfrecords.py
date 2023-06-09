import sys, os
import tensorflow as tf
import numpy as np
from utils import *
import codecs
import re
import unicodedata

"""
This function defines a vocabulary lookup table using two dictionaries char2idx and idx2char. 
The dictionaries map characters to their corresponding indices and indices to their corresponding 
characters respectively.
"""
def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.char_set)}
    idx2char = {idx: char for idx, char in enumerate(hp.char_set)}
    return char2idx, idx2char

"""
 This function takes a string of text as input and normalizes it by removing any diacritical marks or 
 accents, converting it to lowercase, and replacing any characters that are not in a predefined character 
 set with whitespace. The function then returns the normalized text.
"""
def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn') # Strip accents
    text = text.lower()
    text = re.sub("[^{}]".format(hp.char_set), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

"""
This function takes a numerical value as input, and returns it as a TensorFlow Feature object with an int64_list attribute.
"""
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

"""
This function takes a byte string as input, and returns it as a TensorFlow Feature object with a bytes_list attribute
"""
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""
This function encodes the spectrograms and text data into a TensorFlow Example object and writes it to a 
file using a tf.python_io.TFRecordWriter object. 
The function takes three inputs: a writer object to write to a file, the file path of the audio file, 
and the text data. 
The function first loads the spectrograms from the given file path, then encodes them 
along with the text data using the _bytes_feature and _int64_feature functions. 
The resulting Example object is then written to the file using the writer.write() method
"""
def encode_and_write(writer, fpath, text):
    fname, mel, mag = load_spectrograms(fpath)
    example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'mel_raw': _bytes_feature(mel.tostring()),
                        'mag_raw': _bytes_feature(mag.tostring()),
                        'frame_length': _int64_feature(len(mag)),
                        'wav_filename': _bytes_feature(tf.compat.as_bytes(fname)),
                        'text_length': _int64_feature(len(text)),
                        'text': _bytes_feature(np.array(text, dtype=np.float32).tostring())
                    })
              )
    writer.write(example.SerializeToString())
    return

def make_tfrecords(mode):
    # check save dir exists
    if not os.path.exists(hp.feat_path):
        os.makedirs(hp.feat_path)
    
    # load vocabulary
    char2idx, idx2char = load_vocab()

    # parse
    lines = codecs.open(hp.transcript_path, 'r', 'utf-8').readlines()
    # preserving first batch for evaluation
    lines = lines[hp.batch_size:]
    # split tfrecords into small partitions for globally shuffle
    size = len(lines) // hp.tfrecords_partition
    lines_split = [lines[i:i+size] for i in range(0, len(lines), size)]
    for part_idx, lines in enumerate(lines_split):
        # write TFRecords Object
        output_filepath = os.path.join(hp.feat_path,
                    '{}_{}.tfrecords'.format(mode, str(part_idx).zfill(4))
                )
        writer = tf.python_io.TFRecordWriter(output_filepath)
        for line in lines:
            print(line.strip().split('|'))
            fdir, fname, text = line.strip().split('|')
            fpath = os.path.join(hp.data_path, fdir, fname)
            text = text_normalize(text) + "E"  # E: EOS
            text = [char2idx[char] for char in text]
            if not fpath.endswith(".wav"):
                fpath += '.wav'
            encode_and_write(writer, fpath, text)
        writer.close()
    return

def eval_infer_load_data(mode):
    # load vocabulary
    char2idx, idx2char = load_vocab()
    
    # for eval
    if mode == 'eval':
        # parse
        lines = codecs.open(hp.transcript_path, 'r', 'utf-8').readlines()
        # take the batch as eval
        lines = lines[:hp.batch_size]
        fpaths, text_lengths, texts = [], [], []
        for line in lines:
            fdir, fname, text = line.strip().split('|')
            fpath = os.path.join(hp.data_path, fdir, fname)
            text = text_normalize(text) + "E"  # E: EOS
            text = [char2idx[char] for char in text]
            fpaths.append(fpath)
            text_lengths.append(len(text))
            texts.append(text)
        return fpaths, text_lengths, texts

    # for infer
    else:
        lines = codecs.open(hp.infer_data_path, 'r', 'utf-8').readlines()
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines]
        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(sents), maxlen), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def main():
    # extract training feat
    mode = 'train'
    make_tfrecords(mode)

if __name__ == '__main__':
    main()
