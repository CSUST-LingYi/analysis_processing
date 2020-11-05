from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
import pandas as pd


def wordvec_pca(sc):
    corpus = []
    for i, line in sc.iterrows():
        corpus.append(line.fails.strip())
    '''                                                                                              
        2、计算tf-idf设为权重                                                                               
    '''

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    '''                                                                                              
        3、获取词袋模型中的所有词语特征                                                                             
        如果特征数量非常多的情况下可以按照权重降维                                                                        
    '''

    word = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(word)))

    '''                                                                                              
        4、导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示                                                   
    '''
    tfidf_weight = tfidf.toarray()
    print(pd.DataFrame(tfidf_weight))
    '''                                                                                              
        5、PCA主成分分析，数据降维                                                                              
    '''
    pca = PCA(n_components=2)
    decomposition_data = pca.fit_transform(tfidf_weight)
    return decomposition_data