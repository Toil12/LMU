import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import random
import numpy as np
from torch import optim
from torch import autograd

class TaggerModel(nn.Module):
    def __init__(self,
                 numWords:int,
                 numTags: int,
                 embSize: int,
                 rnnSize: int,
                 dropoutRate: float,
                 device:torch.device):
        super(TaggerModel, self).__init__()
        self.device=device
        self.num_layers=1
        self.rnnsize=rnnSize
        self.hidden_dim = 200
        self.word_embeddings = nn.Embedding(numWords, embedding_dim=embSize)

        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(input_size=embSize,
                            hidden_size=self.hidden_dim,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=1,
                            dropout=dropoutRate)
        # 线性层将隐状态空间映射到标注空间
        self.hidden2tag = nn.Linear(rnnSize * 2, numTags)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        h0 = torch.zeros(self.num_layers * 2, sentence.size(0), self.hidden_dim).to(self.device)  # 同样考虑向前层和向后层
        c0 = torch.zeros(self.num_layers * 2, sentence.size(0), self.hidden_dim).to(self.device)
        # sentence=
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), (h0, c0))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print(tag_scores)
        return tag_scores

    def annotate(self, sentence:list)->list:
        tag_scores=self.forward(sentence)
        IDs=[]
        for item in tag_scores:
            max = -1
            max_index = 0
            for index, value in enumerate(item):
                if value > max:
                    max = value
                    max_index = index
            IDs.append(max_index)
        return IDs

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("../model/test/loss.txt", mode='r+', encoding='utf-8') as words_save_file:
        words_save_file.truncate()
    training_data = [([1, 2, 3], [4, 3, 2]),
                     ([2, 3, 3], [3, 2, 2]),
                     ([1, 4, 5, 2], [4, 1, 0, 3])]#([words indexes],[tags])
    validation_data=[([3,2,1],[2,3,4]),
                     ([5,5,5],[0,0,0]),
                     ([1,2,4],[4,3,1])]
    model = TaggerModel(numWords=10,
                        numTags=5,
                        embSize=3,
                        rnnSize=3,
                        device=device,
                        dropoutRate=0.1)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(50):  # 再次说明下, 实际情况下你不会训练300个周期, 此例中我们只是构造了一些假数据
        count = 0
        average_loss = 0
        time_start = time.time()
        print(training_data)
        shuffled_data = np.random.permutation(training_data)
        print(shuffled_data)
        for words, tags in shuffled_data:
            count += 1
            # Step 1\. 请记住 Pytorch 会累加梯度
            # 每次训练前需要清空梯度值
            model.zero_grad()
            # 此外还需要清空 LSTM 的隐状态
            # 将其从上个实例的历史中分离出来
            model.hidden = model.init_hidden()
            # Step 2\. 准备网络输入, 将其变为词索引的 Variables 类型数据
            sentence_in = torch.LongTensor(words)
            targets = torch.LongTensor(tags)
            # Step 3\. 前向传播
            tag_scores = model(sentence_in)
            # print(tag_scores)
            # Step 4\. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
            loss = loss_function(tag_scores, targets)
            average_loss = (average_loss + loss) / count
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f'../model/test/run_test{epoch+1}.pth')
        time_end=time.time()
        print("Episode {} gets loss {}, cost time {} s".format(epoch + 1, average_loss, time_end - time_start))
        with open("../model/test/loss.txt", mode='a', encoding='utf-8') as words_save_file:
            words_save_file.write(str(epoch+1)+' '+str(average_loss.item()))
            words_save_file.write('\n')

    # load model
    model.load_state_dict(torch.load('../model/test/run_test50.pth'))

    # start validation
    for words, tags in validation_data:
        wordIDs = torch.LongTensor(words)
        tagIDs = torch.LongTensor(tags)
        bestTagIDs = model.annotate(wordIDs)
        print(bestTagIDs)


if __name__ == '__main__':
    run_test()
    # with open('../model/test/loss.txt') as f:
    #     for line in f:
    #         print(f)