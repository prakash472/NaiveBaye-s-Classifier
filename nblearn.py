import string
import sys
import os
vocabulary={}
stop_words=set()
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
        characters=[words for words in sentence[0] if words not in string.punctuation]
        cleaned_characters="".join(characters)
        cleaned_data.append([cleaned_characters])
    return cleaned_data

def removeStopWords(data):
    global stop_words
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

# File Path
base_dir="op_spam_training_data"
positive_deceptive_file_path=base_dir+"/positive_polarity/deceptive_from_MTurk/"
positive_truthful_file_path=base_dir+"/positive_polarity/truthful_from_TripAdvisor/"
negative_deceptive_file_path=base_dir+"/negative_polarity/deceptive_from_MTurk/"
negative_truthful_file_path=base_dir+"/negative_polarity/truthful_from_Web/"

def generateTokens(path_1,path_2):
    global vocabulary
    count=0
    tokens_dictionary={}
    file_paths=[path_1,path_2]
    for file_path in file_paths:
        for dir_path,dir_name,file_name in os.walk(file_path):
            for single_file in file_name:
                if not single_file.startswith('.'):
                    file_path=os.path.join(dir_path,single_file)
                    file_contents=data_extraction(file_path)
                    cleaned_file=data_cleaning(file_contents)
                    tokens_dictionary=calculateFeatureDictionary(tokens_dictionary,cleaned_file)
                    vocabulary=calculateFeatureDictionary(vocabulary,cleaned_file)
                    count+=1
    return count, tokens_dictionary

def calculateTotalVocabulary(positive_dict,negative_dict,truthful_dict,deceptive_dict):
    vocabulary={}
    total_dict=[positive_dict,negative_dict,truthful_dict,deceptive_dict]
    for i in range(len(total_dict)):
        for tokens in total_dict[i]:
            if tokens in vocabulary:
                vocabulary[tokens]+=total_dict[i][tokens]
            else:
                vocabulary[tokens]=total_dict[i][tokens]
    return vocabulary

def handleMissingTokens(tokens_dict,vocab_dict):
    final_dict={}
    for token in vocab_dict:
        if token in tokens_dict:
            final_dict[token]=tokens_dict[token]
        else:
            final_dict[token]=0
    return final_dict

def calculateConditionalProbabilities(class_tokens,class_count,vocabulary):
    conditional_prob={}
    class_size=0
    for tokens in class_tokens:
        class_size+=class_tokens[tokens]
    for tokens in class_tokens:
        conditional_prob[tokens]=(class_tokens[tokens]+1)/(len(vocabulary)+class_size)
    return conditional_prob


def generatStopWords(file_path_1,file_path_2):
    def totalTokenCounts(tokens_count,cleaned_file):
        tokens_count=tokens_count
        for tokens in cleaned_file:
            if tokens not in tokens_count:
                tokens_count[tokens]=1
            else:
                tokens_count[tokens]+=1
        return tokens_count
        
    total_token_counts={}
    file_paths=[file_path_1,file_path_2]
    for file_path in file_paths:
        for dir_path,dir_name,file_name in os.walk(file_path):
            for single_file in file_name:
                if not single_file.startswith('.'):
                    file_path=os.path.join(dir_path,single_file)
                    file_contents=data_extraction(file_path)
                    cleaned_file=tokenizeData(file_contents)
                    total_token_counts=totalTokenCounts(total_token_counts,cleaned_file[0])
    stop_words=[]
    for single_token in total_token_counts:
        if total_token_counts[single_token]>150:
            stop_words.append(single_token)
    stop_words=[words.lower() for words in stop_words]
    cleaned_stopwords=[]
    for word in stop_words:
        characters=[words for words in word if words not in string.punctuation]
        cleaned_characters="".join(characters)
        cleaned_stopwords.append(cleaned_characters)
    return stop_words

def generateTotalStopWords(stop_words_positve,stop_words_negative,stop_words_truthful,stop_words_deceptive):
    global stop_words
    total=stop_words
    classes=[stop_words_positve,stop_words_negative,stop_words_truthful,stop_words_deceptive]
    for individual_class in classes:
        for stop_word in individual_class:
            if stop_words=="":
                total.add(stop_word)
    return total
#Calculating stopwords from training data
stop_words_positve=generatStopWords(positive_truthful_file_path,positive_deceptive_file_path)
stop_words_negative=generatStopWords(negative_truthful_file_path,negative_deceptive_file_path)
stop_words_truthful=generatStopWords(positive_truthful_file_path,negative_truthful_file_path)
stop_words_deceptive=generatStopWords(positive_deceptive_file_path,negative_deceptive_file_path)

stop_words=generateTotalStopWords(stop_words_positve,stop_words_negative,stop_words_truthful,stop_words_deceptive)
print("The total stop words is",stop_words)


# Generating for tokens for every class:
positive_count,positive_tokens=generateTokens(positive_truthful_file_path,positive_deceptive_file_path)
negative_count,negative_tokens=generateTokens(negative_truthful_file_path,negative_deceptive_file_path)
truthful_count,truthful_tokens=generateTokens(positive_truthful_file_path,negative_truthful_file_path)
deceptive_count,deceptive_tokens=generateTokens(positive_deceptive_file_path,negative_deceptive_file_path)

#We can calculate the prior probabilty of each class
prior_prob_postive_class=positive_count/(positive_count+negative_count)
prior_prob_negative_class=negative_count/(positive_count+negative_count)
prior_pro_truthful_class=truthful_count/(positive_count+negative_count)
prior_prob_deceptive_class=deceptive_count/(positive_count+negative_count)

# Creating the total vocabulary which could be useful for later laplace smoothing
vocabulary=calculateTotalVocabulary(positive_tokens,negative_tokens,truthful_tokens,deceptive_tokens)


# There might be some classes that do not have all the tokens, let's first add the tokens into the each class
positive_tokens_hadling_missing_tokens=handleMissingTokens(positive_tokens,vocabulary)
negative_tokens_handling_missing_tokens=handleMissingTokens(negative_tokens,vocabulary)
truthful_tokens_handling_missing_tokens=handleMissingTokens(truthful_tokens,vocabulary)
deceptive_tokens_handling_missing_tokens=handleMissingTokens(deceptive_tokens,vocabulary)

# Now calculating the conditional Probabilites for each of the token
cond_prob_positive_class=calculateConditionalProbabilities(positive_tokens_hadling_missing_tokens,positive_count,vocabulary)
cond_prob_negative_class=calculateConditionalProbabilities(negative_tokens_handling_missing_tokens,negative_count,vocabulary)
cond_prob_truthful_class=calculateConditionalProbabilities(truthful_tokens_handling_missing_tokens,truthful_count,vocabulary)
cond_prob_deceitful_class=calculateConditionalProbabilities(deceptive_tokens_handling_missing_tokens,deceptive_count,vocabulary)

# print("Deceitful Check",cond_prob_deceitful_class)


with open("nbmodel.txt","w") as fp:
    fp.write(f"prior_prob_postive_class={prior_prob_postive_class}\n")
    fp.write(f"prior_prob_negative_class={prior_prob_negative_class}\n")
    fp.write(f"prior_pro_truthful_class={prior_pro_truthful_class}\n")
    fp.write(f"prior_prob_deceptive_class={prior_prob_deceptive_class}\n")
    fp.write(f"cond_prob_positive_class={cond_prob_positive_class}\n")
    fp.write(f"cond_prob_negative_class={cond_prob_negative_class}\n")
    fp.write(f"cond_prob_truthful_class={cond_prob_truthful_class}\n")
    fp.write(f"cond_prob_deceitful_class={cond_prob_deceitful_class}\n")
    fp.close()