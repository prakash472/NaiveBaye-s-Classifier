import math
import os
import string
import glob
with open("nbmodel.txt","r") as fp:
    prior_prob_postive_class=eval(fp.readline().split("=")[1])
    prior_prob_negative_class=eval(fp.readline().split("=")[1])
    prior_pro_truthful_class=eval(fp.readline().split("=")[1])
    prior_prob_deceptive_class=eval(fp.readline().split("=")[1])
    cond_prob_positive_class=eval(fp.readline().split("=")[1])
    cond_prob_negative_class=eval(fp.readline().split("=")[1])
    cond_prob_truthful_class=eval(fp.readline().split("=")[1])
    cond_prob_deceitful_class=eval(fp.readline().split("=")[1])
    fp.close()

def tokenizeData(data):
    result=[]
    for sentence in data:
        result.append(sentence[0].split(" "))
    return result
def lowerCased(data):
    result=[]
    for sentence in data:
        result.append([words.lower() for words in sentence])
    return result

def removePunctuation(data):
    cleaned_data=[]
    for sentence in data:
        characters=[words for words in sentence[0] if words not in string.punctuation and words[0].isdigit()==False]
        cleaned_characters="".join(characters)
        cleaned_data.append([cleaned_characters])
    return cleaned_data

def removeStopWords(data):
    # stop_words = ("each","has", "had", "having", "do", "does", "did", "doing", "few", "more", "most", "other", "some", "such", "no","about", "against", "between", "into", "through", "during", "before","i", "me", "my", "myself", "we", "our", "ours","and", "but", "if",  "so", "than", "too", "very", "s", "t", "can", "will", "just", "or", "because", "as", "until", "while", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",  "of", "at", "by", "for", "with",  "after", "above","her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "have",  "a", "an", "the",  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",  "nor",  "am", "is", "are", "was", "were", "be", "been", "being","not", "only", "own", "same","don", "should", "now","")
    stop_words={'just', 'these', 'hadn', 'having', "haven't", "wouldn't", 'o', 'of', 'which', 'your', "needn't", 'be', 'ourselves', 'through', 'too', "mightn't", 'weren', 'me', "shouldn't", 'does', 'himself', "didn't", "doesn't", 'was', 'same', "she's", 'themselves', 'until', 'only', 'own', 'who', 'ain', 'for', 'this', 'ours', 'herself', 'it', 'should', 'that', 'during', 'aren', 'to', 'them', "should've", 'she', 'needn', 'you', "that'll", "you've", "you'll", 'we', 'other', 'm', 'been', 'and', 'all', 'him', 'so', 'an', "shan't", 'yourself', 'itself', 'again', 'not', 'where', 'on', 'such', 'mustn', 'when', 'but', 'now', 'd', 'wasn', 've', 'y', 'by', 'in', "weren't", 'if', 'don', 'is', 'have', 'further', 'hers', 'had', 'doing', 'i', 'nor', 'doesn', 'between', 'my', 'down', 's', 'then', 'shan', 'both', "don't", 'were', 'or', 'no', "won't", 'its', 'their', "wasn't", 'did', 'about', 'with', "aren't", 'our', 'from', "you'd", 'some', 'after', 'under', "it's", 'into', "hasn't", 'didn', 'whom', 'any', 'off', 'before', 'most', 'out', 't', 'each', 'more', 'as', "hadn't", 'yourselves', 'those', 'they', 'how', "you're", 'll', 'there', 'do', 'won', 'the', 'hasn', 'at', 'her', 'than', 'are', 'over', 're', 'couldn', 'will', 'theirs', 'wouldn', 'his', 'ma', "couldn't", 'few', 'what', 'against', 'haven', 'while', 'up', 'once', 'here', 'isn', "isn't", 'mightn', 'myself', 'being', 'he', 'can', 'am', "mustn't", 'yours', 'below', 'above', 'very', 'has', 'shouldn', 'because', 'why', 'a'}
    cleaned_data=[]
    for sentence in data:
        cleaned_data.append([words for words in sentence if words not in stop_words])
    return cleaned_data

def data_cleaning(data):
    lower_cased=lowerCased(data)
    punctuation_cleaned=removePunctuation(lower_cased)
    tokenize_data=tokenizeData(punctuation_cleaned)
    stop_words_cleaned=removeStopWords(tokenize_data)
    return stop_words_cleaned

def data_extraction(file_path):
    result=[]
    with open(file_path,'r') as fp:
        content=fp.read().splitlines()
        result.append(content)
    return result


def calculateFeatureDictionary(dict,data):
    features_dictionary=dict
    for sentence in data:
        for words in sentence:
            if words not in features_dictionary:
                features_dictionary[words]=1
            else:
                features_dictionary[words]+=1
    return features_dictionary

def generateTokens(file_path):
    tokens_dictionary={}
    file_contents=data_extraction(file_path)
    cleaned_file=data_cleaning(file_contents)
    tokens_dictionary=calculateFeatureDictionary(tokens_dictionary,cleaned_file)
    return tokens_dictionary


def calculatePosteriorProb(file_path,class_label):
    global prior_prob_postive_class
    global cond_prob_positive_class
    global prior_prob_negative_class
    global cond_prob_negative_class
    global prior_pro_truthful_class
    global cond_prob_truthful_class
    global prior_prob_deceptive_class
    global cond_prob_deceitful_class

    posterior_prob=0
    tokens_dictionary=generateTokens(file_path)
    # print("Prior prob positive",cond_prob_positive_class)
    # print("Prior prob negative",cond_prob_negative_class)
    # print("Prior prob truthful",cond_prob_truthful_class)
    # print("Prior prob deceitful",cond_prob_deceitful_class)
    if class_label=="positive":
        posterior_prob=math.log(prior_prob_postive_class,10)
        for tokens in tokens_dictionary:
            if tokens not in cond_prob_positive_class:
                continue
            else:
                posterior_prob+=math.log(cond_prob_positive_class[tokens],10)
    elif class_label=="negative":
        posterior_prob=math.log(prior_prob_negative_class,10)
        for tokens in tokens_dictionary:
            if tokens not in cond_prob_negative_class:
                continue
            else:
                posterior_prob+=math.log(cond_prob_negative_class[tokens],10)
    elif class_label=="truthful":
        posterior_prob=math.log(prior_pro_truthful_class,10)
        for tokens in tokens_dictionary:
            if tokens not in cond_prob_truthful_class:
                continue
            else:
                posterior_prob+=math.log(cond_prob_truthful_class[tokens],10)
    else:
        posterior_prob=math.log(prior_prob_deceptive_class,10)
        for tokens in tokens_dictionary:
            if tokens not in cond_prob_deceitful_class:
                continue
            else:
                posterior_prob+=math.log(cond_prob_deceitful_class[tokens],10)
    return posterior_prob
file_pointer=open("nboutput.txt","w")
base_dir="op_spam_testing_data"
testing_files_paths=glob.glob(base_dir+"/*/*/*/*.txt")
for file_path in testing_files_paths:
    positive_posterior_prob=calculatePosteriorProb(file_path,"positive")
    # print("Posterior prob is",positive_posterior_prob)
    negative_posterior_prob=calculatePosteriorProb(file_path,"negative")
    # print("Negative",negative_posterior_prob)
    truthful_posterior_prob=calculatePosteriorProb(file_path,"truthful")
    # print("Truth ful",truthful_posterior_prob)
    deceitful_posterior_prob=calculatePosteriorProb(file_path,"deceptive")
    # print("deceitful",deceitful_posterior_prob)
    # break
    if positive_posterior_prob>negative_posterior_prob:
        label_1="postive"
    else:
        label_1="negative"
    if truthful_posterior_prob>deceitful_posterior_prob:
        label_2="truthful"
    else:
        label_2="deceptive"
    predicted_answer=f"{label_1} {label_2} {file_path} \n"
    file_pointer.write(predicted_answer)
file_pointer.close()


