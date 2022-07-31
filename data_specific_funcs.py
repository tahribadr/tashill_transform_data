import numpy as np

def extract_examples_pautomac(text):
    "takes as input pautomac text file, extracts a list of sequences, each sequence is a list of strings"
    #turn each line into a list of strings, one string for each number.
    lines = text.split('\n')

    #last line of txt file is empty, we remove it
    del lines[-1]

    lines = [line.split(' ') for line in lines]
    #extract list of only sequences
    seq_list = [ line[1:] for line in lines[1:] ]

    return seq_list

def extract_labels_pautomac(text):  
    lines = text.split('\n')
    #last line of txt file is empty, we remove it
    del lines[-1]
    #extract list of only sequences
    labels =  lines[1:] 
    return labels

def extract_HeinzData(text):
    data_list = text.split('\n') 
    data_list = [ line.split('\t')for line in data_list ]
    data_list = data_list[:-1]
    word_list = [line[0] for line in data_list]
    word_list = [ [x for x in row ] for row in word_list ]
    label_list = [line[1] for line in data_list]
    return word_list, label_list


if __name__ == '__main__':
    print("done!")