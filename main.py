from Data import Data
from TaggerModel import TaggerModel
from torch import optim
import numpy as np
import torch
import torch.nn as nn
import math
import time

t_file = "train.tagged"
d_file = "dev.tagged"
numWords = 300
data = Data(train_file=t_file, dev_file=d_file, numWords=numWords)
epochs = 1

def training():
    # Initialization
    with open("model/rnn_model/loss.txt", mode='r+', encoding='utf-8') as words_save_file:
        words_save_file.truncate()
    # do the train
    model = TaggerModel(numWords=numWords+1,
                        numTags=data.numTags+1,
                        embSize=int(math.log(numWords, 2)),
                        rnnSize=1,
                        hiddenSize=int(math.log(numWords, 2)),
                        dropoutRate=0.1)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    # start training
    print("start from {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    for epoch in range(epochs):
        time_start = time.time()
        count = 0
        average_loss = 0
        for words, tags in data.trainSentences:
            count += 1
            wordIDs = data.words2IDs(words)
            tagIDs = data.tags2IDs(tags)
            # print(wordIDs)
            # print(tagIDs)

            # Step 1\. 请记住 Pytorch 会累加梯度
            # 每次训练前需要清空梯度值
            model.zero_grad()

            # 此外还需要清空 LSTM 的隐状态
            # 将其从上个实例的历史中分离出来
            model.hidden = model.init_hidden()

            # Step 2\. 准备网络输入, 将其变为词索引的 Variables 类型数据
            sentence_in = torch.LongTensor(wordIDs)
            targets = torch.LongTensor(tagIDs)

            # Step 3\. 前向传播
            tag_scores = model(sentence_in)
            # print(tag_scores)
            # Step 4\. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
            loss = loss_function(tag_scores, targets)
            average_loss = (average_loss + loss) / count
            loss.backward()
            optimizer.step()
            # print(count)
        torch.save(model.state_dict(), f'model/rnn_model/model{epoch+1}.pth')
        time_end = time.time()
        print("end at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        print("Episode {} gets loss {}, costs {}s".format(epoch + 1, average_loss, time_end - time_start))
        print()
        with open("model/rnn_model/loss.txt", mode='a', encoding='utf-8') as words_save_file:
            words_save_file.write(str(epoch + 1) + ' ' + str(average_loss.item()))
            words_save_file.write('\n')

def predict(model_name):
    # do the train
    model = TaggerModel(numWords=numWords+1,
                        numTags=data.numTags+1,
                        embSize=int(math.log(numWords, 2)),
                        rnnSize=1,
                        hiddenSize=int(math.log(numWords, 2)),
                        dropoutRate=0.1)
    model.load_state_dict(torch.load(model_name))

    # start validation
    corretness=[]
    count = 0
    average_loss = 0
    correct_count = 0
    for words, tags in data.devSentences:
        count+=1
        wordIDs = data.words2IDs(words)
        tagIDs = data.tags2IDs(tags)
        # change to tensor
        wordIDs = torch.LongTensor(wordIDs)
        tagIDs = torch.LongTensor(tagIDs)
        bestTagIDs = model.annotate(wordIDs)
        bestTags = data.IDs2tags(bestTagIDs)


        for i in range(len(bestTags)):
            if bestTags[i]==tags[i]:
                correct_count+=1
        print("this sentence with correctness of {:.4}%".format((correct_count/count*100),'.4f'))
    corretness.append(correct_count/count)
    print("average correctness is {}%".format(np.mean(corretness)*100))




if __name__ == '__main__':
    # training()
    predict('model/rnn_model/model5.pth')