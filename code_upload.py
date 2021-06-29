


import collections
from mxnet import gluon,init,nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata,loss as gloss,nn,rnn,utils as gutils
import os
import random
import tarfile



def read_imdb(folder='train'):
    """Read the IMDB data set for sentiment analysis."""
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('d:/data/aclImdb/', folder, label)
        
        for file in os.listdir(folder_name):
            
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data
train_data,test_data=read_imdb('train'),read_imdb('test')





def get_tokenized_imdb(data):#预处理数据集，对每条评论做分词，基于空格进行分词
    """Get the tokenized IMDB data set for sentiment analysis."""
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]




def get_vocab_imdb(data):#根据分好词的训练数据来创建词典，过滤掉出现次数少于5的词
    """Get the vocab for the IMDB data set for sentiment analysis."""
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5,reserved_tokens=['<pad>'])
vocab = get_vocab_imdb(train_data)
'#words in vocab:',len(vocab)





def preprocess_imdb(data, vocab):#因为每条评论长度不一致，所以不能直接组合成小批量，该函数对每条评论进行分词，并通过字典转换成词索引，
    #通过截断或补<pad>符号来将每条评论长度固定为500
    """Preprocess the IMDB data set for sentiment analysis."""
    max_l = 500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [
            vocab.token_to_idx['<pad>']] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels




#创建数据迭代器，每次迭代将返回一个小批量的数据
batch_size=64
train_set=gdata.ArrayDataset(*preprocess_imdb(train_data,vocab))
test_set=gdata.ArrayDataset(*preprocess_imdb(test_data,vocab))
train_iter=gdata.DataLoader(train_set,batch_size,shuffle=True)
test_iter=gdata.DataLoader(test_set,batch_size)




for X,y in train_iter:
    print('X',X.shape,'y',y.shape)
    break
'#batches:',len(train_iter)





class BiRNN(nn.Block):
    def __init__(self,vocab,embed_size,num_hiddens,num_layers,**kwargs):
        super(BiRNN,self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab),embed_size)
        #bidirectional 设为True即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens,num_layers=num_layers,bidirectional=True,input_size=embed_size)
        self.decoder = nn.Dense(2)
    
    def forward(self,inputs):
        #input的形状是（批量大小，词数），因为LSTM需要将序列作为第一维，所以将输入转置后再提取词特征，输出形状为（词数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        #rnn.LSTM只传入输入embeddings,因此只返回最后一层的隐藏层在各时间步的隐藏状态。outputs形状是（词数，批量大小，2*隐藏单元个数）
        outputs = self.encoder(embeddings)
        #连结初始时间步的隐藏状态作为全连接层输入。它的形状为（批量大小，4*隐藏单元个数）
        encoding = nd.concat(outputs[0],outputs[-1])
        outs = self.decoder(encoding)
        return outs





#创建一个含2个隐藏层的双向循环神经网络
embed_size,num_hiddens,num_layers,ctx=100,100,2,d2l.try_all_gpus()
net = BiRNN(vocab,embed_size,num_hiddens,num_layers)
net.initialize(init.Xavier(),ctx=ctx)





glove_embedding = text.embedding.create('glove',pretrained_file_name='glove.6B.100d.txt',vocabulary=vocab)





net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req','null')


# In[ ]:
import time
import mxnet as mx
from mxnet import autograd
import matplot.pyplot as plt
#训练模型
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

print("begin train")
lr,num_epochs = 0.01,5
trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr})
loss = gloss.SoftmaxCrossEntropyLoss()
def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    """Train and evaluate a model."""
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
       ctx = [ctx]
    print("begin echo……")
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
           
            Xs, ys, batch_size = _get_batch(batch, ctx)   
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))
        x1=range(0,1)
        plt.plot(x1,loss)

train(train_iter,test_iter,net,loss,trainer,ctx,num_epochs)
print("end train")





#定义预测函数
def predict_sentiment(net,vocab,sentence):
    #sentence = nd.array(vocab.to_indices(sentence))
    sentence = nd.array(vocab.to_indices(sentence),ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1,-1))),axis = 1)
    return 'positive' if label.asscalar()==1 else 'negative'





s = predict_sentiment(net,vocab,['this','moive','is','so','great'])
print(s)
t = predict_sentiment(net,vocab,['this','moive','is','so','boring'])
print(t)







