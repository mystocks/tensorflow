#coding=utf-8
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from TrainKDataSet import realKDayData_train
import logging
import time
rootPath = os.path.dirname(os.getcwd())
sys.path.append(rootPath)
logPath = os.path.join(rootPath, 'myrnn.log')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=logPath, filemode='w')

#——————————————————导入数据——————————————————————
#f=open('.\dataset\dataset_1.csv')
#df=pd.read_csv(f)     #读入股票数据
#data=np.array(df['最高价'])   #获取最高价序列
#data=data[::-1]      #反转，使数据按照日期先后顺序排列
#print(data)
#以折线图展示data
#plt.figure()
#plt.plot(data)
#plt.show()
#normalize_data=(data-np.mean(data))/np.std(data)  #标准化
#normalize_data=normalize_data[:,np.newaxis]       #增加维度


class rnn_nton(object):
    #设置常量
    time_step = 60      #时间步
    rnn_unit = 10       #hidden layer units
    batch_size = 60     #每一批次训练多少个样例
    input_size = 1      #输入层维度
    output_size = 1     #输出层维度
    lr = 0.0003         #学习率
    index = 0
    kdata = None
    totallen = 0

    # ——————————————————定义神经网络变量——————————————————
    X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
    Y = tf.placeholder(tf.float32, [None, output_size])  # 每批次tensor对应的标签，这里修改为预测最后一次的涨跌
    # 输入层、输出层权重、偏置
    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }
    def __init__(self):
        self.mgetdata = realKDayData_train()
        self.file_path = os.getcwd()
        self.restore_path = os.path.join(self.file_path, 'module2')
        self.save_path = os.path.join(self.restore_path, 'stock.model')
        pass

    def __getzdf(self, input):
        ret = []
        lenofdata = len(input)
        index = 1
        while index < lenofdata:
            zdf = (10.0*(input[index] - input[index-1]))/input[index-1] # 计算涨跌幅，结果扩大10倍
            ret.append([zdf])
            index += 1
        return ret

    def __getonedata(self):
        data = self.mgetdata.getOneOrgStockData(self.batch_size + self.time_step + 1, 1)
        if data is None:
            return False
        normalize_data = np.array(data[1]['close']) # data[0]代表index
        normalize_data = self.__getzdf(normalize_data)
        totallen = len(normalize_data)
        return True, totallen, normalize_data

    def get_rnn_traindata(self):
        '''
        输入原始股票数据，然后再进行归一化，归一化后再作为训练的输入数据
        :param normalize_data:
        :return:
        '''
        train_x, train_y = [], []  # 训练集
        i = 0
        normalize_data = self.kdata
        while i < self.batch_size:
            if self.index + self.time_step + 1 >self.totallen:
                ret, self.totallen, self.kdata = self.__getonedata()
                if ret == True:
                    normalize_data = self.kdata
                    self.index = 0
                else:
                    return False, train_x, train_y
            temp = normalize_data[self.index:self.index+self.time_step+1]
            x = temp[0:self.time_step]
            y = temp[self.time_step] # 获取最后一次作为结果
            train_x.append(x)
            train_y.append(y)
            self.index += 1
            i += 1
        return True,train_x,train_y



    #——————————————————定义神经网络变量——————————————————
    def lstm(self, batch):      #参数：输入网络批次数目
        w_in = self.weights['in']
        b_in = self.biases['in']
        input = tf.reshape(self.X, [-1, self.input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in)+b_in
        input_rnn = tf.reshape(input_rnn, [-1, self.time_step, self.rnn_unit])  #将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
        init_state = cell.zero_state(batch,dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        #output=tf.reshape(output_rnn,[-1,self.rnn_unit]) #作为输出层的输入
        '''
        这里output_rnn的输出是(batch_size,time_step,rnn_uint),需要转换成(time_step, batch_size,rnn_uint)
        转换后即可直接获取最后一个输出output_rnn[-1]作为预测
        '''
        output = tf.transpose(output_rnn, (1, 0, 2)) # 转换维度，编程(time_step, batch_size,rnn_uint)
        w_out=self.weights['out']
        b_out=self.biases['out']
        pred=tf.matmul(output[-1], w_out)+b_out # 去output最后一只值作为预测
        return pred, final_states



    #——————————————————训练模型——————————————————
    def train_lstm(self):
        pred,_=self.lstm(self.batch_size)
        #损失函数
        loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(self.Y, [-1])))
        train_op=tf.train.AdamOptimizer(self.lr).minimize(loss)
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            module_file = tf.train.latest_checkpoint(self.restore_path)
            if module_file is not None:
                saver.restore(sess, module_file)
            #重复训练10000次
            for step in range(1000000000):
                time.sleep(0.5)
                ret,train_x,train_y = self.get_rnn_traindata()
                if ret == True:
                    _,loss_=sess.run([train_op,loss],feed_dict={self.X:train_x,self.Y:train_y})
                    #ret, trainx, trainy = self.get_rnn_traindata()
                    #每10步保存一次参数
                    if step%99==0 and step != 0:
                        logging.info("step:%d, loss = %f", step,loss_)
                        saver.save(sess, self.save_path)
                    step+=1
                else:
                    break


#train_lstm()


    #————————————————预测模型————————————————————
    def prediction(self):
        pred,_=self.lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            #参数恢复
            module_file = tf.train.latest_checkpoint(self.restore_path)
            saver.restore(sess, module_file)

            #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
            prev_seq=self.train_x[-1]
            predict=[]
            #得到之后100个预测结果
            for i in range(100):
                next_seq=sess.run(pred,feed_dict={self.X:[prev_seq]})
                predict.append(next_seq[-1])
                #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
            #以折线图表示结果

    def get_predict_zd__all(self):
        count_num = 1
        pred,_=self.lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint(self.restore_path)
            saver.restore(sess, module_file)

            stock_list = self.mgetdata.get_all_stock_id_from_network()
            for stockid in stock_list:
                logging.info("%d:start to predict %s", count_num, stockid)
                self.get_predict_zd_one(stockid, self.time_step, sess, pred)
                count_num += 1
                time.sleep(0.5)


    def get_predict_zd_one(self, stockid, time_setp, sess, pred):
        orgdata_df = self.mgetdata.getorg_predictdata_from_datebase(stockid, time_setp)
        if orgdata_df is None:
            logging.warning("getdata is None")
            return
        orgdata_df = self.get_orgdata_zdf(orgdata_df)
        train_input = []
        for index in orgdata_df.index:
            time.sleep(0.1)
            if index < time_setp:
                continue
            cur_predict = orgdata_df.loc[index]['predict1']
            if cur_predict == 99999.9:
                train_input = np.array(orgdata_df.loc[index-time_setp+1:index]['close'])
                train_input = train_input[:,np.newaxis]
                next_seq = sess.run(pred, feed_dict={self.X: [train_input]})
                pre_zdf = next_seq[0][0]/10.0
                if pre_zdf > 0.1:
                    pre_zdf = 0.1
                if pre_zdf < -0.1:
                    pre_zdf = -0.1
                cur_data = orgdata_df.loc[index]['date']
                self.mgetdata.update_itemdata_to_database(stockid, cur_data, pre_zdf)

    def get_orgdata_zdf(self, orgdata_df):
        orgdata_df['newzd']=0.0
        preclose = 0.0
        for index in orgdata_df.index:
            one_data = orgdata_df.loc[index]['close']
            if preclose > 0.0:
                orgdata_df.loc[index, 'newzd'] = (10.0*(one_data - preclose))/preclose
            preclose  = one_data
        return orgdata_df

#prediction()

rnn = rnn_nton()
#rnn.train_lstm()
#rnn.get_predict_zd_one('603999', 100)
rnn.get_predict_zd__all()
logging('predict end')
exit(0)
