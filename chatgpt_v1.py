import random
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
#block = 8
#batch = 4

def readfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        file = txt_file.read()
        length = len(file)
        print(f"\nThis file has {length} characters")
        tokens = []
        for i in range(1, 1000):
            tokens.append(file[i-1])
        print(f"\nThe first 1000 characters are", tokens)
#    return file[:1000]
    return file

def uniqchar(file_content):
    chars = sorted(list(set(file_content)))
    vocab_size = len(chars)
#    print(''.join(chars))
    print(vocab_size)
    return chars

def chartoint(file_content, stoi):
    int_list = encode(file_content, stoi)
#    print(f"\nThe integer list length is :", len(int_list))
    l = len(int_list)
    n = int(0.9 * len(int_list))
    r = round(n)
    train_data = int_list[:r]
    val_data = int_list[r:]
    return train_data, val_data

def getbatch(list):
    origlist = list
    context = []
    target = []
    rand = random.randint(0, 800)
    print(f"\nThe train_data list length is :", len(origlist))
    print(f"\nRandom number generated is :", rand)
    for i in range(rand, (rand + block_size * batch_size), block_size):
        context.append(origlist[i:i + block_size])
        target.append(origlist[i + 1:i + 1 + block_size])
    print(f"\nThe context list is :", context)
    print(f"\nThe target list is :", target)
    return context, target

def decode(int_list, itos):
    """Decoder: Take a list of integers and decode it into a string using the itos mapping"""
    return ''.join([itos[i] for i in int_list])

# Call each function externally
txt_file_path = '/Users/pmarhath/Downloads/Llama/python/chatgpt/kalidasa.txt'
file_content = readfile(txt_file_path)
sort_list = uniqchar(file_content)
print(f"\nvocab_size is ", len(sort_list))

# here are all the unique characters that occur in this text
chars = sorted(list(set(file_content)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s, stoi: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l, itos: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

train, val = chartoint(file_content, stoi)
#print(f"\ntraining data is ", train)
xb, yb = getbatch(train)
print(f"\nxb and yb is ", xb, yb)

# Decode the training data
decoded_train = decode(train, itos)
print("Decoded Training Data:", decoded_train)
