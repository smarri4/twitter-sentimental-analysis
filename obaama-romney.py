import pandas as pd
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
import nltk
from spellcheck import correct
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import time

# from sklearn.model_selection import GridSearchCV

start_time = time.time()

stop = stopwords.words('english')

porter=PorterStemmer()
#config to turn pickle on(True) or off(False)
turn_pickle_on_off = True

doCrossValidation= False

#this function loads the dataframes from the excel sheets for obama and romney
#file_name='train.xlsx'
def load_data(type):
    file_name_train='train1.xlsx'
    file_name_test='testing-Obama-Romney-tweets.xlsx'
    if type=='train':
        obama = pd.read_excel(file_name_train,usecols=[2, 3],sheetname='Obama')
        romney = pd.read_excel(file_name_train,usecols=[2, 3],sheetname='Romney')
    elif type=='test':
        obama = pd.read_excel(file_name_test,usecols=[0, 4],sheetname='Obama')
        romney = pd.read_excel(file_name_test,usecols=[0, 4],sheetname='Romney')
    #assign collect column names
    obama.columns = ['tweets', 'clas']
    romney.columns = ['tweets', 'clas']
    # remove the first row as it is useless: Note that this has effect that the rows start from 1 instead of 0 index.
    obama = obama[obama.index != 0]
    romney = romney[romney.index != 0]
    obama = obama[obama.index != 1]
    romney = romney[romney.index != 1]

    obama=obama[obama.clas.apply(lambda x: isinstance(x, (int, np.int64)))]
    romney=romney[romney.clas.apply(lambda x: isinstance(x, (int, np.int64)))]
    #remove rows that have class as 2.
    obama = obama[obama['clas'] != 2]
    obama = obama.dropna(subset = ['tweets'])
    obama = obama.dropna(subset = ['clas'])#train_df_obama[np.isfinite(train_df_obama['clas'])]
    romney = romney[romney['clas'] != 2]
    romney = romney.dropna(subset = ['tweets'])
    romney = romney.dropna(subset = ['clas'])

    #code to remove skew of 0's
    # obama_having_ones = obama[obama['clas'] == 1]
    # obama_having_minus_one = obama[obama['clas'] == -1]
    # obama=obama.append(obama_having_ones)
    # obama=obama.append(obama_having_minus_one)
    romney_having_ones = romney[romney['clas'] == 1]
    romney_having_minus_one = romney[romney['clas'] == -1]
    romney=romney.append(romney_having_ones)
    romney=romney.append(romney_having_minus_one)
    return obama,romney

#function to clean the tweets
def cleanTweet(tweet):
    tweet_clean=""
    #tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    #remove tags
    tweet = re.sub(r'<.*?>', r'', tweet)
    tweet = re.sub("([a-z])([A-Z])","\g<1> \g<2>",tweet)
    #replace abbr
    tweet = replaceAbbr(tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #remove user handles
    tweet = re.sub('@[^\s]+',r'',tweet)
    #remove links from the tweets
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',r'',tweet)
    #remove non-english characters
    tweet = re.sub(r'[^a-zA-Z]', r' ', tweet)
    words = tweet.strip().split(" ")
    #stem each word to find its root using porter stemmer
    for each_word in words:
        if(each_word.isalpha()):
            each_word=correct(each_word) #spell check
            if(each_word.strip() != ""):
                each_word = re.sub('\W+','', each_word)
                each_word = porter.stem(each_word)
                if each_word not in stop:
                    tweet_clean = tweet_clean + each_word.strip() + " "
    tweet_clean=tweet_clean.strip()
    return tweet_clean

abbr_dict = {}
def readAbbrFile(abbrFile):
    global abbr_dict

    f = open(abbrFile)
    lines = f.readlines()
    f.close()
    for i in lines:
        tmp = i.split('|')
        abbr_dict[tmp[0]] = tmp[1]
    return abbr_dict
# This function checks the dictionary containing abbreviations and their meanings as (key,value) pairs
# and replaces the key with the corresponding value =
abbrFile = 'replace.txt'
abbr_dict = readAbbrFile(abbrFile)
def replaceAbbr(s):
    for word in s:
        if word.lower() in abbr_dict.keys():
            s = [abbr_dict[word.lower()] if word.lower() in abbr_dict.keys() else word for word in s]
    return s
#end

def build_pilelines(classifiers):
    # tweet_corpus = []
    pipelines = []
    # for tweet in data_frame['tweets']:
    #     tweet_corpus.append(str(tweet))
    for classifier in classifiers:
        text_clf = Pipeline([('vect', CountVectorizer(input='content',stop_words='english', ngram_range=(1,2))),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier),
        ])
        pipelines.append(text_clf)
    return pipelines#,tweet_corpus


if __name__=='__main__':
    if not turn_pickle_on_off:
        print('Loading the data..')
        obama,romney=load_data('train')
        obama_act_test,romney_act_test = load_data('test')
        obama['clas']=obama['clas'].apply(int)
        romney['clas']=romney['clas'].apply(int)
        obama_act_test['clas']=obama_act_test['clas'].apply(int)
        romney_act_test['clas']=romney_act_test['clas'].apply(int)
        print('data loaded successfully..')
        print('cleaning tweets..')
        #clean the tweets
        tweetCleaner=lambda x: cleanTweet(str(x))
        obama['tweets']=obama['tweets'].apply(tweetCleaner)
        romney['tweets']=romney['tweets'].apply(tweetCleaner)
        obama_act_test['tweets']=obama_act_test['tweets'].apply(tweetCleaner)
        romney_act_test['tweets']=romney_act_test['tweets'].apply(tweetCleaner)
        print('tweets cleaned!')
        #fetching cleaned tweets from pickled file
        output1 = open('obama.pkl', 'wb')
        output2 = open('romney.pkl', 'wb')
        output3 = open('obama_test.pkl', 'wb')
        output4 = open('romney_test.pkl', 'wb')
        pickle.dump(obama,output1)
        pickle.dump(romney,output2)
        pickle.dump(obama_act_test,output3)
        pickle.dump(romney_act_test,output4)
    print('loading obama and romney dataframes with cleaned tweets from pickled file..')
    pkl_file1 = open('obama.pkl', 'rb')
    obama = pickle.load(pkl_file1)
    pkl_file2 = open('romney.pkl', 'rb')
    romney = pickle.load(pkl_file2)
    pkl_file3 = open('obama_test.pkl', 'rb')
    obama_act_test = pickle.load(pkl_file3)
    pkl_file4 = open('romney_test.pkl', 'rb')
    romney_act_test = pickle.load(pkl_file4)

    tweet_corpus_obama = []
    tweet_corpus_romney = []
    for tweet in obama['tweets']:
         tweet_corpus_obama.extend(str(tweet).split())

    tweet_corpus_romney = []
    for tweet in romney['tweets']:
         tweet_corpus_romney.extend(str(tweet).split())
    #----   OBAMA   -------------------------------------------------------------------------------------------------
    folds = 10
    # cross validation
    ss = ShuffleSplit(n_splits=folds, test_size=0.25, random_state=0)
    # classifiers = [MultinomialNB(), SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42), svm.SVC(kernel='linear', C=1), RandomForestClassifier(), AdaBoostClassifier(n_estimators=100), LogisticRegression()]
    classifiers = [MultinomialNB(), SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42), svm.SVC(kernel='linear', C=1), LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier(n_estimators=100)]
    mean_obama = 0
    k=1
    obama=obama.reset_index()
    pipelines = build_pilelines(classifiers)
    for text_clf in pipelines:
        if doCrossValidation:
            for train_index, test_index in ss.split(obama):
                print('fold: ', k)
                k+=1
                #---obama--------------------------------------------------------------------------------------------------
                obama_train, obama_test = obama.ix[train_index], obama.ix[test_index]
                obama_train=obama_train.reset_index()
                obama_test=obama_test.reset_index()
                target_train, target_test = obama['clas'].ix[train_index], obama['clas'].ix[test_index]
                target_train=target_train.reset_index()
                target_test=target_test.reset_index()
                trgt_names = list(set(target_test['clas']))
                trgt_names = [str(i) for i in trgt_names]
        #--------cross fold logic ends here-------------------------------------------------------------------------------

        #--------train/test obama dataframe-------------------------------------------------------------------------------

            #-----Grid Search-------------------------------------------------------
                parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                    'tfidf__use_idf': (True, False),
                    # 'clf__alpha': (1e-2, 1e-3),
                }

                gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
                print(text_clf.named_steps['clf'])
                text_clf.named_steps['vect'].fit(tweet_corpus_obama)
                gs_clf = gs_clf.fit(obama_train['tweets'].values.astype('U'), list(target_train['clas']))
                pipe_predicted = gs_clf.predict(obama_test['tweets'].values.astype('U'))
                print('prediction accuracy: ', np.mean(pipe_predicted == target_test['clas']))
                mean_obama+=np.mean(pipe_predicted == target_test['clas'])
                print(metrics.classification_report(list(target_test['clas']), pipe_predicted, target_names=trgt_names))

                #---commenting out original train and predict to use grid search
               #  print(text_clf.named_steps['clf'])
               #  text_clf.named_steps['vect'].fit(tweet_corpus)
               # # print(text_clf.named_steps['vect'].get_feature_names())
               #  print('obama: \n')
               #  # obama_train = obama_train.sort(columns = 'tweets')
               #  text_clf = text_clf.fit(obama_train['tweets'].values.astype('U'), list(target_train['clas']))
               #  pipe_predicted = text_clf.predict(obama_test['tweets'].values.astype('U'))
               #  print('prediction accuracy: ', np.mean(pipe_predicted == target_test['clas']))
               #  mean_obama+=np.mean(pipe_predicted == target_test['clas'])
               #  print(metrics.classification_report(list(target_test['clas']), pipe_predicted, target_names=trgt_names))

            k=1
            print('overall prediction accuracy for obama: ', mean_obama/folds)
            # print('overall prediction accuracy for romney: ', mean_romney/folds)
            mean_obama=0

    folds = 10
    # cross validation
    ss = ShuffleSplit(n_splits=folds, test_size=0.25, random_state=0)
    # classifiers = [MultinomialNB(), SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42), svm.SVC(kernel='linear', C=1), RandomForestClassifier(), AdaBoostClassifier(n_estimators=100)]
    classifiers = [svm.SVC(kernel='linear', C=1)]
    mean_romney = 0
    k=1
    romney=romney.reset_index()
    flag=0
    for text_clf in pipelines:
        if doCrossValidation:
            for train_index, test_index in ss.split(romney):
                print('fold: ', k)
                k+=1
                #----romney-------------------------------------------------------------------------------------------------
                romney_train, romney_test = romney.ix[train_index], romney.ix[test_index]
                romney_train=romney_train.reset_index()
                romney_test=romney_test.reset_index()
                target_train_romney, target_test_romney = romney['clas'].ix[train_index], romney['clas'].ix[test_index]
                target_train_romney=target_train_romney.reset_index()
                target_test_romney=target_test_romney.reset_index()
                trgt_names = list(set(target_test_romney['clas']))
                trgt_names = [str(i) for i in trgt_names]
        #--------cross fold logic ends here-------------------------------------------------------------------------------

        #--------train/test obama dataframe-------------------------------------------------------------------------------

            #-----Grid Search-------------------------------------------------------
                parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)],
                              'vect__max_features':[3000],
                    'tfidf__use_idf': (True, False),
                    # 'clf__alpha': (1e-2, 1e-3),
                }

                gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
                print(text_clf.named_steps['clf'])
                text_clf.named_steps['vect'].fit(tweet_corpus_romney)
                if flag==0:
                    print(text_clf.named_steps['vect'].get_feature_names())
                    flag=1
                gs_clf = gs_clf.fit(romney_train['tweets'].values.astype('U'), list(target_train_romney['clas']))
                pipe_predicted = gs_clf.predict(romney_test['tweets'].values.astype('U'))
                print('prediction accuracy: ', np.mean(pipe_predicted == target_test_romney['clas']))
                mean_romney+=np.mean(pipe_predicted == target_test_romney['clas'])
                print(metrics.classification_report(list(target_test_romney['clas']), pipe_predicted, target_names=trgt_names))

            k=1
            print('overall prediction accuracy for romney: ', mean_romney/folds)
            mean_romney=0

        print(text_clf.named_steps['clf'])
        text_clf = text_clf.fit(obama['tweets'].values.astype('U'), list(obama['clas']))
        predicted_obama = text_clf.predict(obama_act_test['tweets'].values.astype('U'))
        print('prediction accuracy of obama on test data: ', np.mean(predicted_obama == obama_act_test['clas']))
        print(metrics.classification_report(list(obama_act_test['clas']), predicted_obama))

        text_clf = text_clf.fit(romney['tweets'].values.astype('U'), list(romney['clas']))
        predicted_romney = text_clf.predict(romney_act_test['tweets'].values.astype('U'))
        print('prediction accuracy of romney on test data: ', np.mean(predicted_romney == romney_act_test['clas']))
        print(metrics.classification_report(list(romney_act_test['clas']), predicted_romney))

    print("program execution took %s seconds ---" % (time.time() - start_time))


