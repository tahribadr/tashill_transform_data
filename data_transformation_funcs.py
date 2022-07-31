import random
from tkinter import N 
import numpy as np
import os

## Useful functions to transform pautomac into a shuffled agreed upon standard

def generate_vocab(seq_list):
    """ouputs a list containing the vocabulary of the sequence/words list"""
    vocab = set()
    for seq in seq_list:
        vocab.update(set(seq)) 
    vocab = list(vocab)
    vocab.sort()
    return vocab

def generate_label_vocab(label_list):
    """ouputs a list containing the vocabulary of the label list"""
    return list(set(label_list))

# takes as input a list of sequences, each sequence is a list of letters. 
# outputs a randomized mapping dictionary for all letters to an integer
def generate_random_mapping(seq_list, mode):
    if mode =="words":
        vocab = generate_vocab(seq_list)
    elif mode =="labels":
        vocab = generate_label_vocab(seq_list)

    else:
        print("the mode argument can either be \"labels\" or \"words\"")
    random_encodings = random.sample( list( range( len(vocab) ) ), len(vocab) )
    mapping_dict = dict( zip( vocab, random_encodings ) )
    return mapping_dict 

#takes as input a list of sequence lists and encodes them using the mapping dictionary
def map_list(mapping_dict, data, print_debug=False):
    data_copie = [line.copy() for line in data]
    if print_debug:
        print(data_copie[-1], len(data_copie))
    for index,seq in enumerate(data_copie):
        if print_debug and index<10:
            print(seq)
        for letter_index,letter in  enumerate(seq) :
            if print_debug:
                print("word index", index, "letter index", letter_index, "letter", letter)
            data_copie[index][letter_index] = mapping_dict[letter] 
            
    return data_copie

def map_labels(mapping_dict, data, print_debug=False):
    data_copie = data.copy()
    for letter_index,letter in  enumerate(data_copie) :
        if print_debug and letter_index<10:
            print("letter index", letter_index, "letter", letter, "mapping_dict[letter] ", mapping_dict[letter])
        data_copie[letter_index] = mapping_dict[letter]
    return data_copie

def shuffle(seq_list, seq_labels= None):
    """inputs list, returns shuffled list and indices array, that will be useful to unshuffle
       if labels are provided, shuffles the labels in the same way as the sequences """
    seq_list = np.array(seq_list,dtype=object)
    indices = np.arange(len(seq_list))
    random.shuffle(indices)
    shuffled_array = seq_list[indices]
    #turn all nested sequences into  a list as thet were in the input
    shuffled_list = [ list( word ) for word in  shuffled_array]

    if seq_labels != None:
        seq_labels = np.array(seq_labels)
        shuffled_labels = list( seq_labels[indices] )
        return shuffled_list,shuffled_labels, indices
    return shuffled_list, indices


def reverse_shuffle(shuffled_list, shuffle_indices, shuffled_labels= None):
    """inputs shuffled array and shuffle_indices array, returns ordered list
       if labels are provided, unshuffles the labels in the same way as the sequences """

    shuffled_array = np.array(shuffled_list,dtype=object)
    reverse_indices = np.array( [ int( np.where(shuffle_indices == i)[0] ) for i in range( len(shuffled_array) ) ] )
    ordered_array = shuffled_array[reverse_indices]
    ordered_list = [ list( word ) for word in  ordered_array]

    if shuffled_labels != None:
        shuffled_labels = np.array(shuffled_labels)
        ordered_labels = list( shuffled_labels[reverse_indices] )
        return ordered_list, ordered_labels

    return ordered_list

def reverse_shuffle_all(train_list, valid_list, test_list, test_labels, dataset_key, task,
                        train_labels = None,
                        valid_labels = None):
    if task == "lm":
        train_list = reverse_shuffle(train_list, dataset_key["train_key"])
        valid_list = reverse_shuffle(valid_list, dataset_key["valid_key"])

    elif task == "classif":
        train_list,train_labels = reverse_shuffle(train_list,
                                                shuffled_labels=train_labels,  
                                                shuffle_indices=dataset_key["train_key"])
        valid_list,valid_labels = reverse_shuffle(valid_list,
                                                shuffled_labels=valid_labels,  
                                                shuffle_indices=dataset_key["valid_key"])

    test_list,test_labels = reverse_shuffle(test_list,
                                            shuffled_labels=test_labels,  
                                            shuffle_indices=dataset_key["test_key"])
    
    if task == "lm" :
        return train_list, valid_list, test_list, test_labels
    elif task == "classif":
        return train_list, train_labels, valid_list, valid_labels, test_list, test_labels


def add_beg_end(train_list, vocab_size):
    # vocab_size = len( generate_vocab(train_list) )
    #make a copy so as not to change the original list
    train_list_copy = train_list.copy()
    for index in range( len(train_list_copy) ) :
        train_list_copy[index].insert(0, vocab_size)
        train_list_copy[index].append( vocab_size+1 )
    return train_list_copy

def transform_lists(train_list, test_list, test_labels,
                       valid_list = None, 
                       train_labels = None,
                       valid_labels = None,
                       valid_ratio = 10, 
                       categorical = False,
                       task = "lm"):

    #generate random mapping and transform data
    word_mapping = generate_random_mapping(train_list, mode="words")

    #transform data using the mapping
    train_list = map_list(word_mapping, train_list)
    test_list = map_list(word_mapping, test_list)

    # case when categorical, gotta generate vocab and assign a mapping to all_labels
    # case when task = classif, we need to prepare valid and train labels as well
    if (not categorical) and task == "lm":
        test_labels = [float(score) for score in test_labels]

    elif (not categorical) and task == "classif":
        train_labels = [float(score) for score in train_labels]
        valid_labels = [float(score) for score in valid_labels]
        test_labels = [float(score) for score in test_labels]

    elif categorical and task == "lm":
        label_mapping = generate_random_mapping(test_labels, mode = "labels")
        test_labels = map_labels(label_mapping, test_labels)

    elif categorical and task == "classif":
        label_mapping = generate_random_mapping(train_labels, mode = "labels")

        train_labels = map_labels(label_mapping, train_labels)
        valid_labels = map_labels(label_mapping, valid_labels)
        test_labels = map_labels(label_mapping, test_labels)

    #generate a vocabulary list from the training set
    vocab = generate_vocab(train_list)
    vocab_size = len(vocab)

    train_list = add_beg_end(train_list, vocab_size)
    test_list = add_beg_end(test_list, vocab_size)


    # if no validation list is provided, we carve it out of the train list
    if valid_list == None :
 
        #little math trick to get a number of train examples divisible by valid_ratio
        #we might ignore a few examples at the end of the dataset for this to work
        div_factor = 1/(valid_ratio+1)
        nr_train_examples = int((1-div_factor)*len(train_list)/valid_ratio)*valid_ratio
        # here, we will need original train_list and train labels to remain the same for both slices, 
        # so we copy it
        old_train_list = train_list.copy()
        train_list = old_train_list[:nr_train_examples]
        valid_list = old_train_list[nr_train_examples:nr_train_examples+(nr_train_examples//valid_ratio)]

    else :
        #we map the valid list if provided
        valid_list = map_list(word_mapping, valid_list)
        #if valid list is provided but train list is too big, we cut examples from train list
        if len(train_list) >= valid_ratio*len(valid_list):
            train_list = train_list[:valid_ratio*len(valid_list)]

        #else, we adapt train list to be dividable by valid ratio and we cut examples from valid list
        else :
            dividable_by_valid_ratio = int( len(train_list)/valid_ratio )*valid_ratio
            train_list = train_list[:dividable_by_valid_ratio]
            valid_list = valid_list[:len(train_list)/valid_ratio]

        #we only add begin end if we didn't carve out valid set from training
        valid_list = add_beg_end(valid_list, vocab_size)

 
        

    #shuffle sequences randomly to make it harder for competitors to guess the dataset
    if task == "lm":
        train_list,train_key = shuffle(train_list)
        valid_list,valid_key = shuffle(valid_list)
    elif task == "classif":
        train_list, train_labels, train_key = shuffle(train_list, train_labels)

        valid_list, valid_labels, valid_key = shuffle(valid_list, valid_labels)

    test_list, test_labels, test_key = shuffle(test_list, test_labels)

    dataset_key = {"train_key":train_key,
                    "valid_key":valid_key,
                    "test_key":test_key,
                    "word_mapping": word_mapping}
    if categorical :
        dataset_key["label_mapping"] = label_mapping

    if task == "lm":
        return train_list, valid_list, test_list, test_labels, dataset_key
    elif task == "classif":
        return train_list, train_labels, valid_list, valid_labels, test_list, test_labels, dataset_key


def generate_data_text(data_list):
    """   
    returns a string variable of the data in pautomac format

    Params:

    data_list: list of sequences, each sequence is a list of stringifiable objects
    Keyword arguments:
    vocab_size: size of vocabulary of the data
    """ 
    # here we work on a copy so as not to modify data_list, we will need it later 
    # WATCH OUT, this is a nested list, need to copy all lists inside !!
    data_list_copy = [data.copy() for data in data_list]
    nr_examples = len(data_list_copy)
    vocab = set()
    
    for index,seq in enumerate(data_list_copy):
        vocab.update(set(seq)) 
        seq.insert(0, len(seq))
        #turn all elemnts of sequences into string
        data_list_copy[index] = [str(letter)for letter in seq ]

    vocab_size = len(vocab)

    data_list_copy.insert(0,[str(nr_examples),str(vocab_size)])

    
    #turn each line back to a single string with numbers separated by a space
    lines = [' '.join(line) for line in data_list_copy]
    #our new data text is 
    text = '\n'.join(lines)

    return text

def generate_label_text(label_list, categorical):
    """returns a string var of the labels in pautomac format
    Params:
    label_list: list of stringifiable objects
    categorical: is a boolean, true if labels are categorical"""

    #here we work on a copy so as not to modify data_list, we will need it later 
    label_list_copy = label_list.copy()

    #turn all elemnts of sequences into string
    label_list_copy= [str(label)for label in label_list_copy ]

    if categorical:
        num_classes = len(set(label_list_copy)) 
        label_list_copy.insert(0,[str(len(label_list_copy)),str(num_classes)])

    else:
        label_list_copy.insert(0,[str(len(label_list_copy))])
    #turn each line back to a single string with numbers separated by nothing
    lines = [''.join(line) for line in label_list_copy[1:]]
    #except for the first line, where we separate nr of examples and nr of classes by a space
    lines.insert(0, ' '.join(label_list_copy[0]))
    #our new data text is 
    text = '\n'.join(lines)

    return text

def make_competition_sets(train_list, 
                        valid_list, 
                        test_list, 
                        test_labels,
                        data_id ,
                        categorical,
                        train_labels = None,
                        valid_labels = None,
                        task = "lm",
                        target_path = os.getcwd()):

    competition_name = "tashill"


    train_text = generate_data_text(train_list)
    valid_text = generate_data_text(valid_list)
    test_text = generate_data_text(test_list)

    if task == "classif" :
        train_labels_text = generate_label_text(train_labels, categorical = categorical)
        valid_labels_text = generate_label_text(valid_labels, categorical = categorical)

    test_labels_text = generate_label_text(test_labels, categorical = categorical)

    with open(os.path.join(target_path,str(data_id)+'.'+competition_name+'.train.words'), 'w') as f:
        f.write(train_text)
    with open(os.path.join(target_path,str(data_id)+'.'+competition_name+'.valid.words'), 'w') as f:
        f.write(valid_text)
    with open(os.path.join(target_path,str(data_id)+'.'+competition_name+'.test.words'), 'w') as f:
        f.write(test_text)
    with open(os.path.join(target_path,str(data_id)+'.'+competition_name+'.test.labels'), 'w') as f:
        f.write(test_labels_text)
    if task == "classif":
        with open(os.path.join(target_path,str(data_id)+'.'+competition_name+'.train.labels'), 'w') as f:
            f.write(train_labels_text)
        with open(os.path.join(target_path,str(data_id)+'.'+competition_name+'.valid.labels'), 'w') as f:
            f.write(valid_labels_text)

    return

def reverse_beg_end(train_list):
    train_list = [line[1:-1] for line in train_list]
    return train_list

def reverse_map(mapping_dict, data, print_debug=False):

    reverse_mapping_dict = {val: key for key, val in mapping_dict.items()}

    data_copy = data.copy()
    if print_debug:
        print(data_copy[-1], len(data_copy))
    for index,seq in enumerate(data_copy):
        if print_debug:
            print(seq)
        for letter_index,letter in  enumerate(seq) :
            if print_debug:
                print("word index", index, "letter index", letter_index, "letter", letter)
            data_copy[index][letter_index] = reverse_mapping_dict[letter] 

    return data_copy

def reverse_map_labels(mapping_dict, data):
    reverse_mapping_dict = {val: key for key, val in mapping_dict.items()}
    data_copy = data.copy()

    for index,label in enumerate(data_copy):
        data_copy[index] = reverse_mapping_dict[label] 

    return data_copy

def reverse_transform(train_list, valid_list, test_list, test_labels, dataset_key,task, categorical,
                      train_labels = None,
                      valid_labels = None):
    
    if task == "lm":
        (train_list,
         valid_list, 
         test_list, 
         test_labels) = reverse_shuffle_all(train_list, 
                                            valid_list, 
                                            test_list, 
                                            test_labels, 
                                            dataset_key,
                                            task)
                                                                        
    elif task == "classif":
        (train_list, 
         train_labels, 
         valid_list, 
         valid_labels, 
         test_list, 
         test_labels) = reverse_shuffle_all(train_list, 
                                            valid_list, 
                                            test_list, 
                                            test_labels, 
                                            dataset_key,
                                            task,
                                            train_labels,
                                            valid_labels)
                                    
    word_mapping = dataset_key["word_mapping"]
    if categorical:
        label_mapping = dataset_key["label_mapping"]
        test_labels = reverse_map_labels(label_mapping, test_labels)
        if task == "classif":
            train_labels = reverse_map_labels(label_mapping, train_labels)
            valid_labels = reverse_map_labels(label_mapping, valid_labels)

    train_list, valid_list, test_list = reverse_beg_end(train_list),reverse_beg_end(valid_list),reverse_beg_end(test_list)
    train_list, valid_list, test_list = reverse_map(word_mapping,train_list),reverse_map(word_mapping,valid_list),reverse_map(word_mapping,test_list)

    if task == "lm":
        return train_list, valid_list, test_list,test_labels
    
    elif task == "classif":
        return train_list, train_labels, valid_list, valid_labels, test_list, test_labels
    


if __name__ == '__main__':
    print("done!")