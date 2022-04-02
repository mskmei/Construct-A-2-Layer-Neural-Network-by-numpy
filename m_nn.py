import numpy as np

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    max_col = np.max(x,axis = 1, keepdims = True) #求每列的最大值，并保留位数特性
    exp = np.exp(x - max_col)
    return exp/np.sum(exp, axis = 1, keepdims = True)

def jiaochashang(w1,w2,lambda_,out_of_softmax):         # 此函数计算交叉熵，输入为softmax函数的输出
    c,k = out_of_softmax.shape                          # 事实上c=batch size
    choice_log = -np.log(out_of_softmax[range(c-1),k-1])       # 根据交叉熵的原理，仅有匹配的项目可以相加，其余为0
    loss_1 = 0.5*lambda_*(np.sum(w1*w1)+np.sum(w2*w2))  # L2正则项提供的loss
    loss_2 = np.sum(choice_log)/c      
    return loss_1+loss_2

def acc_cal(x,y):
    acc = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            acc += 1/len(x)
    return(acc)

def save(nn, path):
    import joblib
    joblib.dump(nn, path)

class NN(object):
    # class nn 应当实现：神经网络结构构造，求解loss
    def __init__(self, input_size, hidden, output, small = 1e-5):
        self.para = {}
        self.para["w1"] = small*np.random.randn(input_size,hidden)
        self.para["b1"] = small*np.zeros((1,hidden))
        self.para["w2"] = small*np.random.randn(hidden,output)
        self.para["b2"] = small*np.zeros((1,output))
        self.lr = 0
        self.hid = 100
        self.reg = 0
        #随机初始化w和b

    def calculate_loss_gradient(self, X, y, lambda_):
        # X(batch size, input size),对于X的每一行都是一个被拉长的图片
        # y(1, batch size)

        w1,b1,w2,b2 = self.para['w1'],self.para['b1'],self.para['w2'],self.para['b2']
        N = X.shape[0] #记X(N,in)

        h = relu(np.dot(X,w1)+b1) # 隐藏层 relu(X*w1+b)   (N,hidden)
        out = np.dot(h,w2) + b2  #此处为array的加法，对每一行加b2(而每一行均代表一个被拉长的图片 (N,out)
        probabilities = softmax(out)    #最终经过softmax的输出
        loss = jiaochashang(w1,w2,lambda_,probabilities)   #交叉熵算得的loss
        
        gradient = {}
        probabilities[range(N), y] -= 1
        djiaochashang = (probabilities)/N # 交叉熵对Z(即上面的out)的导数为A-Y(Y为单位阵)
        dout = djiaochashang #只是为了表示简便

        #有了交叉熵函数的导数，我们下面进行backpropagation步骤
        dw2 = np.dot(h.T,dout) #(hidden,out)
        gradient["w2"] = dw2+lambda_*w2
        db2 = np.sum(dout,axis = 0, keepdims = True)
        gradient["b2"] = db2
        dh = np.dot(dout,w2.T) # (N,hidden)
        dh[h <= 0] = 0
        dw1 = np.dot(X.T,dh)  #(in,hidden)
        gradient["w1"] = dw1+lambda_*w1
        db1 = np.sum(dh,axis = 0,keepdims = True)
        gradient["b1"] = db1
        return loss, gradient

    def predict(self, X):
        prediction = None
        probabilities = relu(np.dot(X,self.para['w1'])+self.para['b1'])
        probabilities = np.dot(probabilities,self.para['w2'])+self.para['b2']
        prediction = np.argmax(probabilities, axis = 1)
        return prediction

    
    def train(self,X,y,X_val,y_val,lr = 0.001,lr_decrease = 0.9,lambda_ = 0, 
                epochs = 20, batch_size = 16, if_print = True):
        sum_num = X.shape[0]          # 总个数，即X的第一维度。因为X每一行均为一个被拉长的矩阵。
        iterations_per_epoch = sum_num//batch_size
        if iterations_per_epoch<1:
            iterations_per_epoch = 1 #保证每个epoch的迭代次数至少一次
        sum_iter= epochs * iterations_per_epoch

        loss_storage = []
        train_acc_storage = []
        val_acc_storage = []
        for iteration in range(1,sum_iter):
            sample = np.random.choice(sum_num, batch_size, replace = True)
            # 上一步进行了随机抽样，随机抽取了batch_size个数（允许重复），从而实现SGD
            X_sample = X[sample,:]
            y_sample = y[sample]
            #对X，y对应sample部分进行了选取，用于本一轮的实验。可以看做是对batch个图片标签训练
            #其中，X每一行均为一个图片矩阵拉长的向量
            loss,gradient = self.calculate_loss_gradient(X_sample,y_sample,lambda_ = lambda_)
            

            # 进行梯度下降处理
            self.para['w2'] -= lr * gradient['w2']
            self.para['b2'] -= lr * gradient['b2']
            self.para['w1'] -= lr * gradient['w1']
            self.para['b1'] -= lr * gradient['b1']

            if not (iteration % iterations_per_epoch):
                # 每当进行完一个epoch进行学习率调整
                epoches = iteration / iterations_per_epoch
                train_acc = acc_cal(self.predict(X_sample),y_sample)
                val_acc = acc_cal(self.predict(X_val),y_val)
                train_acc_storage.append(train_acc)
                val_acc_storage.append(val_acc)
                loss_storage.append(loss)
                # print(self.predict(X_val),y_val)
                if if_print:
                    print('epoches: %d, sum_epoches :%d, train acc: %f, validation acc = %f, loss = %f'%
                                (epoches, sum_num, train_acc, val_acc, loss))
                if len(loss_storage)>=2 and loss_storage[-1]>loss_storage[-2]:
                    lr *= lr_decrease #学习率更新策略
        return loss_storage,train_acc_storage,val_acc_storage
