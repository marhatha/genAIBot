import random
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

block = 8
batch = 4

def readfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        file = txt_file.read()
        length = len(file)
        print(f"\nThis file has {length} characters")
        tokens = []
        for i in range(1,1000):
            tokens.append(file[i-1])
        print(f"\nThe first 1000 characters are", tokens)
 #   return file[:1000]
        return file

def uniqchar(file_content):
# here are all the unique characters that occur in this text
    chars = sorted(list(set(file_content)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)
    return chars

def chartoint(file_content):
    int_list = []
    stoi = {}
    itos = {}
    for i, char in enumerate(file_content):
        int_list.append(i)
        stoi[char] = i
        itos[i] = char
    int_list.pop(0)
    print(f"\nThe integer list length is: {len(int_list)}")
    l = len(int_list)
    n = int(0.9 * len(int_list))
    r = round(n)
    train_data = int_list[:r]
    val_data = int_list[r:]
    return train_data, val_data, stoi, itos

def decode(int_list, itos):
    """Decoder: Take a list of integers and decode it into a string using the itos mapping"""
    return ''.join([itos[char] for char in int_list])

def getbatch(list):
    #block = 8
    #batch = 4
    origlist = list
    context = []
    target = []
    rand = random.randint(0, 800)
    print(f"\nThe train_data list length is :", len(origlist))
    print(f"\nRandom number generated is :", rand)
    for i in range(rand, (rand + block * batch), block):
        context.append(origlist[i:i + block])
        target.append(origlist[i + 1:i + 1 + block])
    print(f"\nThe context list is :", context)
    print(f"\nThe target list is :", target)
    return context,target

def decode(int_list, itos):
    """Decoder: Take a list of integers and decode it into a string using the itos mapping"""
    return ''.join([itos[char] for char in int_list])

# Call each function externally
txt_file_path = '/Users/pmarhath/Downloads/Llama/python/chatgpt/kalidasa.txt'
file_content = readfile(txt_file_path)
sort_list = uniqchar(file_content)
print(f"\nvocab_size is ", len(sort_list))
train, val, stoi, itos = chartoint(file_content)
print(f"\ntraining data is ", train)
xb, yb = getbatch(train)
xb_tensor = torch.tensor(xb)
yb_tensor = torch.tensor(yb)
print(f"\nxb and yb is ", xb_tensor, yb_tensor)

# Decode the training and validation data
decoded_train = decode(train, itos)
decoded_val = decode(val, itos)

print("Decoded Training Data:", decoded_train)
print("Decoded Validation Data:", decoded_val)




