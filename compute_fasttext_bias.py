import gzip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FastText_Bias():

    def __init__(self, domains):
        self.weat_file_path = "dataset/en_weat_file.txt"
        self.embedding_file_path = "../dataset_corpus/fasttext/cc.en.300.vec.gz"
        self.domains = domains
        
    def load_embeddings(self, vocab_words):
        """
        self.word_list = pickle.load(open("dataset/1990-vocab.pkl", "rb"))
        self.word_dic = dict({(x, i) for (i,x) in enumerate(self.word_list)})
        word_vec = np.load("dataset/1990-w.npy")
        
        self.embeddings = {}
        for i in range(len(self.word_list)):
            self.embeddings[self.word_list[i]] = word_vec[i]
            
        print("Embedding Loaded: ", len(self.word_list), len(self.embeddings))
        print()
        return
        """
        self.embeddings = {}
        self.word_list = vocab_words
        self.word_dic = dict({(x, i) for (i,x) in enumerate(self.word_list)})
        
        file_read = gzip.open(self.embedding_file_path, "rt")
        
        token_count = 0
        for line in file_read:
            tokens = line.strip().split(" ")
            token_count += 1
            
            if tokens[0] in self.word_dic:
                #print("Embeddings Found: ", tokens[0])
                self.embeddings[tokens[0]] = [float(x) for x in tokens[1:]]
            
        print("Embedding Loaded: ", token_count, len(self.embeddings))
        print()

    def load_weat_words(self, female_topic="WEAT_Topic_Female", male_topic="WEAT_Topic_Male"):
        file_read = open(self.weat_file_path, "r")
        topic_dict = {}

        print("WEAT Dataset Loading")

        for line in file_read:
            data = line.strip().split(", ")
            current_topic = data[0]

            if current_topic in self.domains:
                topic_dict[current_topic] = [x.lower() for x in data[1:]]
                print(current_topic, topic_dict[current_topic])

        self.female_domain = [female_topic] + topic_dict[female_topic]
        self.male_domain = [male_topic] + topic_dict[male_topic]

        del topic_dict[female_topic]
        del topic_dict[male_topic]
        self.domain_dict = topic_dict
        
    def analyze_roget_words(self, word_list):
        for domain in self.domain_dict:
            self.domain_dict[domain] = word_list       
    
    def average_similarity_wordVSlist(self, word_one, given_list):
        similarity = []
              
        for word_two in given_list: #["lesbian"]:
            if not word_two in self.word_dic:
                continue
              
            #print("Word Check: ", word_two)
            try:
                vec_one = np.array(self.embeddings[word_one])
                vec_two = np.array(self.embeddings[word_two])
                #print("Check Word Index: ", word_dic[word_one], word_dic[word_two])
                #print(vec_one, vec_two)
            except:
                #print(1800+year, word_two, "Does Not Exist!")
                continue

            #print(word_one, word_two, cosine_similarity([vec_one], [vec_two]))

            sim = cosine_similarity([vec_one], [vec_two])
            similarity.append(sim[0][0])
              
        wordsim = np.average(similarity)
        
        #print(word_one, given_list[-1], wordsim)
        return wordsim 
    
    def return_gender_association(self, gender_list):
        domain_association = {}
        
        for domain in self.domain_dict:
            association = []
            for word in gender_list:
                domain_word_list = self.domain_dict[domain]
                word_sim = self.average_similarity_wordVSlist(word, domain_word_list)
                association.append(word_sim)
                
            domain_association[domain] = np.average(association)
        
        return domain_association
                
    def create_data_store_stats(self):

        self.data_store = {}
        self.data_store[self.female_domain[0]] = self.return_gender_association(self.female_domain[1:])
        self.data_store[self.male_domain[0]] = self.return_gender_association(self.male_domain[1:])

        bias_dict = {}
        for domain in self.domain_dict:
            bias_dict[domain] = self.data_store[self.male_domain[0]][domain] - self.data_store[self.female_domain[0]][domain]
            
        return bias_dict

