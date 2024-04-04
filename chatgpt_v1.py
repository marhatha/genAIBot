import random
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


def readfile(file_path):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        file = txt_file.read()
        length = len(file)
        print(f"\nThis file has {length} characters")
        tokens = []
        for i in range(1,1000):
            tokens.append(file[i-1])
        print(f"\nThe first 1000 characters are", tokens)
    return file[:1000]
#    return file

def uniqchar(file_content):
# here are all the unique characters that occur in this text
    list1 = file_content
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
    return len(sort_list)

def chartoint(file_content):
    int_list = []
    for char in file_content:
        int_list.append(ord(char))
    int_list.pop(0)
    print(f"\nThe integer list length is :", len(int_list))
    l = len(int_list)
    n = int(0.9 * len(int_list))
    r = round(n)
    train_data = int_list[:r]
    val_data = int_list[r:]
    return train_data

def getbatch(list):
    block = 8
    batch = 4
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

# Call each function externally
txt_file_path = '/Users/pmarhath/Downloads/Llama/python/chatgpt/kalidasa.txt'
file_content = readfile(txt_file_path)
vocab_size = uniqchar(file_content)
#print(f"\nvocab_size is ",vocab_size)
train = chartoint(file_content)
print(f"\ntraining data is ",train)
xb,yb = getbatch(train)
xb_tensor = torch.tensor(xb)
yb_tensor = torch.tensor(yb)
#print(f"\nxb and yb is ",xb_tensor ,yb_tensor)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        print(f"\nidx inside generate func is ", idx)
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            print(f"\nlogits inside generate func is ",logits)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            print(f"\nlogits after converting to B times C is ", logits)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            print(f"\nidex_next is ",idx_next)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

vocab_size = uniqchar(file_content)
print(f"\nvocab_size is ",vocab_size)

m = BigramLanguageModel(125)
logits, loss = m(xb_tensor, yb_tensor)
print(logits.shape)
print(loss)

# create a PyTorch optimizer

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
total_loss = 0.0  # Accumulate total loss
total_batches = 0  # Count total number of batches

for steps in range(5000):
    xb, yb = getbatch(train)
    xb_tensor = torch.tensor(xb)
    yb_tensor = torch.tensor(yb)
    
    # Evaluate the loss
    logits, loss = m(xb_tensor, yb_tensor)
    total_loss += loss.item()  # Accumulate loss for current batch
    total_batches += 1  # Increment batch count

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

mean_loss = total_loss / total_batches  # Calculate mean loss
print("Mean loss:", mean_loss)


#idx_output = m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()
#print(f"\nidx_output is ",idx_output)
