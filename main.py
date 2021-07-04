import sys
from src.rnn_train import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_file = "data/train.tagged"
d_file = "data/dev.tagged"
numWords = 10000
data = Data(train_file=t_file, dev_file=d_file, numWords=numWords)
epochs =1
embed_zise=100
dropout=0.4
hidden_size=200

if __name__ == '__main__':
    command=sys.argv[1]
    if command=="train":
        training(data=data,
                 numWords=numWords,
                 epochs=epochs,
                 embedding_size=embed_zise,
                 dropout=dropout,
                 hidden_size=hidden_size,
                 device=device)
    elif command=="dev":
        model_name=sys.argv[2]
        predict(f'model/rnn_model/{model_name}.pth',
                data=data,
                numWords=numWords,
                embedding_size=embed_zise,
                dropout=dropout,
                hidden_size=hidden_size,
                device=device)