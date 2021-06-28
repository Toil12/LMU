class Data:
    def __init__(self, train_file: str, dev_file: str, numWords: int):
        """
        Store the data and give out two list related to
        :param train_file:
        :param dev_file:
        :param numWords:
        """
        self.words_frequency_records = {}
        self.words_most_frequent_records = {}
        self.tag_frequency_records = {}
        self.trainSentences = []
        self.devSentences = []
        self.numTags = 1
        self.numWords = numWords
        # read validation file
        with open(dev_file, encoding='utf-8') as f:
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
        with open(train_file, encoding="utf-8") as f:
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
    def store(self, word_path='data/words_index.txt', tag_path='data/tags_index.txt'):
        with open(word_path, 'w', encoding='utf-8') as words_save_file:
            for k, v in self.words_most_frequent_records.items():
                words_save_file.write(k + ' ' + str(v))
                words_save_file.write('\n')

        with open(tag_path, 'w', encoding='utf-8') as tag_save_file:
            for k, v in self.tag_frequency_records.items():
                tag_save_file.write(k + ' ' + str(v))
                tag_save_file.write('\n')

def run_test():
    t_file = "train.tagged"
    d_file = "dev.tagged"
    data = Data(train_file=t_file, dev_file=d_file, numWords=100)
    count=0
    for words, tags in data.trainSentences:
        wordIDs = data.words2IDs(words)
        tagIDs = data.tags2IDs(tags)
        count+=1
    print(tags)
    print(tagIDs)
    print(data.IDs2tags(tagIDs))
    print(count)


if __name__ == '__main__':
    run_test()