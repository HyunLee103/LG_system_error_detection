import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해
import tqdm

def tf(t, d):
    return d.count(t)

def idf(N,t,errtype_doc):
    df = 0
    for doc in errtype_doc:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d,N,errtype_doc):
    return tf(t,d)* idf(N,t,errtype_doc)

def mk_tfidf_feature(df,min,max):
    # df : train_err, test_err
    # min: 회원 id 최솟값
    # count : 회원 수
    errtype_doc =[]
    for id in range(min,max+1):
        errtype_list = list(df[df['user_id'] == id]['errtype'])
        errtype_list = list(map(str, errtype_list))
        errtype_doc.append(errtype_list)

    N = len(errtype_doc) # 총 문서의 수

    vocab = df.errtype.unique()

    result = []
    for i in tqdm(range(N)):
        result.append([])
        d = errtype_doc[i]
        for j in range(len(vocab)):
            t = vocab[j]
            result[-1].append(tfidf(t,d,N,errtype_doc))

    tfidf_ = pd.DataFrame(result, columns = vocab)
    return tfidf_

PATH = '/content/drive/MyDrive/dacon/'

train_err_data = pd.read_csv(PATH+ 'train_err_data.csv')
train_user_id_max = 25000
train_user_id_min = 10000
train_tfidf_= mk_tfidf_feature(train_err_data,10000,25000)
train_tfidf_.to_csv('./train_tfidf.csv')


test_user_id_max = 44999
test_user_id_min = 30000

test_err_data = pd.read_csv(PATH+'test_err_data.csv')
test_tfidf_= mk_tfidf_feature(test_err_data,test_user_id_min,test_user_id_max)
test_tfidf_.to_csv('./test_tfidf.csv')