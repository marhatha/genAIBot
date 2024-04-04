import random
import torch
import numpy as np
def readfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        file = txt_file.read()
        length = len(file)
        print(f"\nThis file has {length} characters")
        tokens = []
        for i in range(1,1000):
            tokens.append(file[i-1])
#        print(f"\nThe first 1000 characters are", tokens)
#        uniqchar(file[:1000])
        uniqchar(file)

def uniqchar(list):
    list1 = list
    check_list = []
    sort_list = []
    for i in range(len(list1)):
        if list1[i]  in check_list:
            continue
        else:
            sort_list.append(list1[i])
            check_list.append(list1[i])
    sort_list.pop(0)
    print(f"\nThe uniq characters in the list are:", sort_list)
    print(f"\nThe vocab size of the sorted list is :", len(sort_list))
    chartoint(list1)

def chartoint(list):
    int_list = []
    char_list = list
    for i in range(len(char_list)):
        int_list.append(ord(char_list[i]))
    int_list.pop(0)
    print(f"\nThe integer list length is :", len(int_list))
    l = len(int_list)
    n = int(0.9*len(int_list))
    r = round(n)
    train_data = int_list[:r]
    val_data = int_list[r:]
#    print(f"Train data set is :", train_data)
#    print(f"Validation data set is :", val_data)
    getbatch(train_data)

def getbatch(list):
    block = 8
    batch = 4
    origlist = list
    context = []
    target = []
#    rand = 100
    rand = random.randint(0,800)
    print(f"\nThe train_data list length is :", len(origlist))
    print(f"\nRandom number generated is :", rand)
    for i in range(rand,(rand+block*batch),block):
        context.append(origlist[i:i+block])
        target.append(origlist[i+1:i+1+block])
    print(f"\nThe context list is :", context)
    print(f"\nThe target list is :", target)
txt_file_path='/Users/pmarhath/Downloads/Llama/python/chatgpt/kalidasa.txt'
readfile(txt_file_path)
