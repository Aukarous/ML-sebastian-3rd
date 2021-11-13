"""
用Latent Dirichlet Allocation实现主题建模
Decomposing text documents with Latent Dirichlet Allocation
主题建模（topic modeling）描述了为无标签文本文档分配主题的任务，目标是为蚊帐指定分类标签，可看作是聚类任务，属于无监督学习的子类
LDA涉及许多数学知识，包括贝叶斯推断
    LDA是生成概率模型，试图找出经常出现在不同文档中的单词。
    假设每个文档都是由不同单词组成的混合体，那些经常出现的单词就代表主题。
    LDA的输入是词袋模型。LDA把词袋模型作为输入，然后分解为两个新矩阵：文档主题矩阵和单词主题矩阵。
    LDA必须手动定义的一个超参数：主题数量
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

if __name__ == "__main__":
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    # 👇 用CountVectorizer类创建词袋矩阵以作为LDA的输入，max_df：要考虑单词的最大文档频率，max_features: 要考虑单词的数量限制，均可调优
    count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
    X = count.fit_transform(df['review'].values)
    lda = LatentDirichletAllocation(n_components=10,
                                    random_state=123,
                                    learning_method='batch',
                                    n_jobs=-1
                                    )
    X_topics = lda.fit_transform(X)
    print('lda components_.shape is {}'.format(lda.components_.shape))

    n_top_words = 5
    feature_names = count.get_feature_names()

    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" %(topic_idx+1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))