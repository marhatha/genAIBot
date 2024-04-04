import random

block = 8
batch = 4

def readfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        file = txt_file.read()
        length = len(file)
        print(f"\nThis file has {length} characters")
        tokens = []
        for i in range(1, 1000):
            tokens.append(file[i-1])
        print(f"\nThe first 1000 characters are", tokens)
    return file[:1000]

def uniqchar(file_content):
    chars = sorted(list(set(file_content)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)
    return chars


def chartoint(file_content, stoi):
    int_list = encode(file_content, stoi)
    print(f"\nThe integer list length is :", len(int_list))
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
    for i in range(rand, (rand + block * batch), block):
        context.append(origlist[i:i + block])
        target.append(origlist[i + 1:i + 1 + block])
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
print(f"\ntraining data is ", train)
xb, yb = getbatch(train)
print(f"\nxb and yb is ", xb, yb)

# Decode the training data
decoded_train = decode(train, itos)
print("Decoded Training Data:", decoded_train)
