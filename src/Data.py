import os

root=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

print(root)
class Data:
    def __init__(self, *args):#train_file: str, dev_file: str, numWords: int

        self.words_frequency_records = {}
        self.words_most_frequent_records = {}
        self.tag_frequency_records = {}
        self.trainSentences = []
        self.devSentences = []
        if len(args)==1:
            self.init_test(*args)
        else:
            self.init_train(*args)

        # read validation file
    def init_test(self,paramfile:str):
        self.paramfile=paramfile

    def init_train(self,train_file: str, dev_file: str, numWords: int, paramfile:str):

        self.train_file = train_file
        self.dev_file = dev_file
        self.numWords = numWords
        self.numTags = 1
        self.paramfile=paramfile
        self.read_data()


    def read_data(self):
        with open(self.dev_file, encoding='utf-8') as f:
            dev_word = []
            dev_tag = []
            for line in f:
                if line.strip():
                    word, tag = line.split()
                    dev_word.append(word)
                    dev_tag.append(tag)
                else:
                    self.devSentences.append((dev_word, dev_tag))
                    dev_word = []
                    dev_tag = []
        # read train file
        with open(self.train_file, encoding="utf-8") as f:
            train_word = []
            train_tag = []
            for line in f:
                if line.strip():
                    word, tag = line.split()
                    train_word.append(word)
                    train_tag.append(tag)
                    # record the frequency
                    if word not in self.words_frequency_records.keys():
                        self.words_frequency_records[word] = 1
                    else:
                        self.words_frequency_records[word] += 1
                    if tag not in self.tag_frequency_records.keys():
                        self.tag_frequency_records[tag] = self.numTags
                        self.numTags += 1
                    else:
                        pass
                else:
                    self.trainSentences.append((train_word, train_tag))
                    train_word = []
                    train_tag = []
        words_n_most_list = sorted(self.words_frequency_records.items(),
                                   key=lambda kv: (kv[1], kv[0]),
                                   reverse=True)
        words_n_most_list = words_n_most_list[0:self.numWords]
        for index, value in enumerate(words_n_most_list):
            if value[0] not in self.words_most_frequent_records.keys():
                self.words_most_frequent_records[value[0]] = index + 1
            else:
                pass

    # record the frequency of words
    def words2IDs(self, words: list) -> list:
        word_indexes = []
        for word in words:
            if word in self.words_most_frequent_records.keys():
                word_indexes.append(self.words_most_frequent_records[word])
            else:
                word_indexes.append(0)
        return word_indexes

    # record the frequency of tags
    def tags2IDs(self, tags: list) -> list:
        tag_indexes = []
        for tag in tags:
            if tag in self.tag_frequency_records.keys():
                tag_indexes.append(self.tag_frequency_records[tag])
            else:
                tag_indexes.append(0)
        return tag_indexes

    def IDs2tags(self, indexes: list) -> list:
        tags=[]
        for index in indexes:
            for k, v in self.tag_frequency_records.items():
                if v == index:
                    tags.append(k)
        return tags

    # store the two index related as files
    def store(self, word_path=f'{root}/data/words_index.txt', tag_path='../data/tags_index.txt'):
        with open(word_path, 'w', encoding='utf-8') as words_save_file:
            for k, v in self.words_most_frequent_records.items():
                words_save_file.write(k + ' ' + str(v))
                words_save_file.write('\n')

        with open(tag_path, 'w', encoding='utf-8') as tag_save_file:
            for k, v in self.tag_frequency_records.items():
                tag_save_file.write(k + ' ' + str(v))
                tag_save_file.write('\n')

    # def store_parameters(self):

def run_test():
    t_file = "../data/train.tagged"
    d_file = "../data/dev.tagged"
    data = Data(t_file, d_file, 10000,"")
    count=0
    for words, tags in data.trainSentences:
        wordIDs = data.words2IDs(words)
        tagIDs = data.tags2IDs(tags)
        count+=1
    print(tags)
    print(tagIDs)
    print(data.IDs2tags(tagIDs))
    print(count)
    data.store()


if __name__ == '__main__':
    run_test()