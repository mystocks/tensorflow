#coding=utf-8
import os
import sys
rootPath = os.path.dirname(os.getcwd())
sys.path.append(rootPath)
import time
from sqlalchemy import create_engine
import tushare as ts
import pymysql as MySQLdb
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import traceback
import logging
logPath = os.path.join(rootPath, 'tensorflow.log')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=logPath, filemode='w')
class realKDayData_train(object):
    tableNameOfStockId = "AllStockId"
    mConnectDb = None
    mCur = None

    MAX_NUM_MEMORY = 100
    mTrainLen = 0 # 用于记录有多少个缓存stock日线data
    mCurIndex = 0 # 表示当前读到缓存中哪个位置了
    mOrgTrainData = [] # 元素item=(index, orgStockData)
    all_stock_Id = []
    def __init__(self):
        self.bInited = False
        self.mConnectDb = MySQLdb.connect(host='127.0.0.1', user='root', passwd='Root@123')
        if self.mConnectDb != None:
            self.mCur = self.mConnectDb.cursor()
            self.mConnectDb.select_db('stocksdb')
            print("connected to MySQLDb!")

    def get_one_stock_data_toDb_byEngine(self, stockId):
        '''
        存储股票数据到数据库，通过pandas的特殊方法
        '''
        try:
            df = ts.get_k_data(stockId)
            #SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://.....'
            engine = create_engine('mysql+pymysql://root:Root@123@127.0.0.1/stocksdb?charset=utf8')
            # 存入数据库
            df.to_sql(stockId+'_KDay', engine, if_exists='replace')
        except:
            logging.error('one stock toDb:\n%s' % traceback.format_exc())
            pass

    def get_all_stock_data_toDb_byEngine(self):
        '''
        获取单只股票近3年日线数据
        '''
        isOk=False
        index=3
        print("Enter get_all_stock_data_toDb_byEngine")
        self.get_all_stock_id_from_network()
        for stockId in self.all_stock_Id:
            print("start to get data ",stockId)
            index=3
            while(index>0):
                try:
                    self.get_one_stock_data_toDb_byEngine(stockId)
                    index=0
                except:
                    index -= 1
                time.sleep(0.5)
        print("Leaver get_all_stock_data_toDb_byEngine")

    def update_one_stock_toDb_bySql(self, stockId):
        '''
        更新日线数据到数据库
        '''
        updateDays = 200
        sqlcmd="SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='%s_KDay';"%stockId
        count = self.mCur.execute(sqlcmd)
        #print "****Start id=", stockId
        if count == 0:
            print("    Not exist table:%s_KDay"%stockId)
            return
        else:
            sqlline="select * from %s_KDay;"%(stockId)
            dataCount=self.mCur.execute(sqlline)
            if dataCount == 0:
                #print "    table is empty"
                return
        valueList=[]

        df = ts.get_k_data(stockId)
        count=len(df)
        indexId=0
        if count <=2:
            #print "    no data from ts"
            return

        if count > updateDays:
            indexId = count - updateDays
            count = updateDays
            data=df[-updateDays:]
        else:
            data=df
        #print data
        va=data.values

        for index in range(count):
            oneData=va[index]
            curDate=oneData[0]
            sqlline="select * from %s_KDay where date='%s';"%(stockId,curDate)
            dateCount=self.mCur.execute(sqlline)
            if dateCount >=1:
                indexId = indexId + 1
                continue
            valueList.append((indexId,oneData[0],oneData[1],oneData[2],oneData[3],oneData[4],oneData[5],oneData[6]))
            indexId = indexId + 1
        #print valueList
        if len(valueList) > 0:
            #sqlline="insert into %s_KDay(date,open,close,high,low,volume,code) values(%%s,%%s,%%s,%%s,%%s,%%s,%%s);"%(stockId)
            sqlline = "insert into %s_KDay values(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s);" % (stockId)
            try:
                pass
                self.mCur.executemany(sqlline,valueList)
                self.mConnectDb.commit()
            except:
                self.mConnectDb.rollback()
            #print "    Done id=",stockId
        else:
            #print "    Done,no data need to insert id=",stockId
            pass

    def get_all_stock_id_from_database(self):
        '''
        从数据库读取所有股票代码
        '''
        self.all_stock_Id = []
        try:
            sqlcmd="SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='%s';"%self.tableNameOfStockId
            count = self.mCur.execute(sqlcmd)
            if count == 0:
                print("Error:Not found table =",self.tableNameOfStockId)
                return
        except:
            print("except..")
        cmdline="select stockId from %s"%self.tableNameOfStockId
        count=self.mCur.execute(cmdline)
        if count == 0:
            print("Error:no data in table=",self.tableNameOfStockId)
        self.all_stock_Id=self.mCur.fetchall()
        print("Suc:get stocksId, the total number is ",len(self.all_stock_Id))
        return self.all_stock_Id

    def store_stock_id_to_db_BySql(self):
        '''
        存储所有股票id到数据库
        :return:
        '''
        stockIdList = self.__get_all_stock_id_from_network()
        r1=None
        try:
            sqlcmd="SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='%s';"%self.tableNameOfStockId
            count = self.mCur.execute(sqlcmd)
            if count == 0:
                cmdline="create table %s(id int,stockId varchar(10))"%self.tableNameOfStockId
                print(cmdline)
                self.mCur.execute(cmdline)
            else:
                print("Exist..")
                #return

            valuses=[]
            i=1
            print(len(self.all_stock_Id))
            for id in self.all_stock_Id:
                valuses.append((i,id))
                i=i+1

            print("valuse len=%d"%len(valuses))
            #这里的%需要使用%%转义字符来格式化
            #在sql里无论是什么类型，都使用%s来作为占位符
            cmdline="insert into %s values(%%s,%%s)"%self.tableNameOfStockId
            print(cmdline,valuses[0])
            r1=self.mCur.executemany(cmdline, valuses)
            print(r1)
            self.mConnectDb.commit()
        except:
            print("except...",r1)
            self.mConnectDb.rollback()

    def set_table_date_unique(self):
        '''
        设置表项属性的唯一属性
        '''
        try:
            for stockId in self.all_stock_Id:
                pass
                sqlcmd = "ALTER TABLE '%s_KDay' ADD unique('date');"%stockId
                print(sqlcmd)
                self.mCur.execute(sqlcmd)
                self.mConnectDb.commit()
        except:
            print("ERROR:ADD unique failed")
            self.mConnectDb.rollback()


    def get_one_data_AndCheck_from_databases(self, stockId, subCount):
        '''
        遍历单个股票数据并过滤目标股票ID保存
        '''
        sqlcmd="SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='%s_KDay';"%stockId
        count = self.mCur.execute(sqlcmd)
        if count == 0:
            print("Not exist table:%s_KDay"%stockId)
            return False
        sqlcmd = "select * from %s_KDay"%stockId
        #param = stockId+"_KDay"
        count=self.mCur.execute(sqlcmd)
        if count <=90:
            return False
        offset = count - 90
        count = 15-3
        self.mCur.scroll(offset, mode='absolute')
        results = self.mCur.fetchall()
        for i in range(count):
            row = i+75
            increaseRate=0.01
            open1=results[row][2]
            open2=results[row+1][2]
            open3=results[row+2][2]
            close1=results[row][3]
            close2 = results[row+1][3]
            close3 = results[row+2][3]
            vol1 = results[row][6]
            vol2 = results[row+1][6]
            vol3 = results[row+2][6]
            #print results[row][2], results[row][3],results[row+1][2], results[row+1][3], results[row+2][2], results[row+2][3]
            if close2 > (close1+close1*increaseRate) and  close3 > (close2+close2*increaseRate):
                if vol2 > (vol1+vol1*increaseRate) and  vol3 > (vol2+vol2*increaseRate):
                    for j in range(subCount):
                        closex=results[row-j][3]
                        if close3 < closex:
                            return False
                    #print stockId,results[row][1],close1
                    return True

    def get_All_data_from_databases_AndCheck(self, functionName, subcount):
        '''
        遍历所有股票数据并过滤目标股票ID保存
        '''
        self.get_all_stock_id_from_database()
        rList=[]
        curCount=0
        allCount=len(self.all_stock_Id)
        for stockId in self.all_stock_Id:
            if True == self.get_one_data_AndCheck_from_databases(stockId, subcount):
                rList.append(stockId)
            if functionName != None:
                functionName(curCount, allCount)
            curCount+=1
        return rList

    def get_all_stock_id_from_network(self):
    # 获取所有股票代码
        try:
            stock_info=ts.get_stock_basics()
            for i in stock_info.index:
                self.all_stock_Id.append(i)
            #print(len(self.all_stock_Id))
        except:
            print('result is:\n%s' % traceback.format_exc())

    def get_all_stockId(self):
        filePathName = os.path.join(rootPath, "data", "AllStockId.csv")
        if os.path.exists(filePathName):
            ret = pd.read_csv(filePathName, encoding='utf8')
            idsArr = np.array(ret['stockIds'])
            ids = idsArr.tolist()
            return ids
        else:
            self.get_all_stock_id_from_network()
            if len(self.all_stock_Id) != 0:
                c = {"stockIds": self.all_stock_Id}
                data = DF(c)
                data.to_csv(filePathName, encoding='utf8')
                return self.all_stock_Id

    def update_all_data_to_db_bySql(self, functionName=None):
        '''
        更新所有日线数据到数据库
        :return:
        '''
        curCount=1
        allCount=0
        #self.get_all_stock_id_from_database()
        retIds = self.get_all_stockId()
        print(retIds)
        #allCount=len(self.all_stock_Id)
        for item in retIds:
            # 这里的stockId是一个tuple
            stockId = "%s"%item
            if len(stockId) < 6:
                left = 6-len(stockId)
                str = ""
                while left > 0:
                    str += "0"
                    left -= 1
                stockId = str + stockId
            try:
                self.update_one_stock_toDb_bySql(stockId)
                curCount += 1
            except:
                print('updata KDay Data Failed:\n%s' % traceback.format_exc())
            if functionName != None:
                functionName(curCount, allCount)
            print("update suc count = ", curCount)
            time.sleep(0.1)

    def check_Exist_StockTable_InDB(self,stockId):
        '''
        判断对应股票表是否存在于数据库中
        :param stockId:
        :return True or False:
        '''
        sqlcmd="SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='%s_KDay';"%stockId
        logging.info(sqlcmd)
        count = self.mCur.execute(sqlcmd)
        if count == 0:
            logging.warning("Not exist table:%s_KDay"%stockId)
            return False
        return True

    def __del__(self):
        self.mConnectDb.close()
        self.mCur.close()

    def get_one_data_form_databases(self, stockId):
        '''
        从数据库读取所有股票信息，并返回结果List
        :param stockId:
        :return:
        '''
        if False == self.check_Exist_StockTable_InDB(stockId):
            #return None
            self.get_one_stock_data_toDb_byEngine(stockId)

        retList=[]
        try:
            cmdline="select date,open,close,high,low,volume from %s_KDay;"%stockId
            logging.info('cmdline = %s', cmdline)
            count=self.mCur.execute(cmdline)
            if count == 0:
                return None
            retList=self.mCur.fetchall()
        except:
            logging.warning('get_one_data_form_databases:\n%s' % traceback.format_exc())
        if len(retList) >0:
            return retList
        return None

    def int2StrStockId(self, stockId):
        strId = "%s" % stockId
        if len(strId) < 6:
            left = 6 - len(strId)
            str = ""
            while left > 0:
                str += "0"
                left -= 1
            strId = str + strId
        return strId

    def get_all_sotck_data_from_database(self):
        retIds = self.get_all_stockId()
        print(retIds)
        trainData = []
        for stockId in retIds:
            if type(stockId) == type(123):
                stockId = self.int2StrStockId(stockId)
            data = self.get_one_data_form_databases(stockId)
            trainData.append(data)

    def readdata_tomemory(self, kcount, labelcount):
        '''
        读取数据到内存中并保存
        1、如果stock_id列表中为空，则主动获取列表
        2、从股票列表弹出一个股票Id，从数据库读取股票的日线数据
        3、把股票数据添加到训练缓存数据列表中
        :param kcount:
        :param labelcount:
        :return:
        '''
        if len(self.all_stock_Id) == 0:
            self.get_all_stock_id_from_network()
            if len(self.all_stock_Id) == 0:
                print("get stockId from Network failed")
                return False
        dataLen = len(self.all_stock_Id)
        for i in range(dataLen):
            stockid = self.all_stock_Id.pop(0)  # 获取第一个元素并弹出
            #print("trainStockId = ", stockid)
            oneData = self.get_one_data_form_databases(stockid)
            if oneData == None or len(oneData) < (kcount + 50):
                continue
            self.mTrainLen += 1
            oneDataDf = pd.DataFrame(np.array(oneData), columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            dfcolmus = ['open', 'close', 'high', 'low', 'volume']
            oneDataDf[dfcolmus] = oneDataDf[dfcolmus].astype('float64')
            self.mOrgTrainData.append([0, oneDataDf])
            if self.mTrainLen > self.MAX_NUM_MEMORY:
                break

    def getOneOrgStockData(self, kcount, labelcount):
        '''
        功能：返回一条数据，包含训练和label，满足kcount和label的要求，并负责更新缓存数据
        :param kCount:
        :param labelcount
        :return traindata, labeldata:
        '''
        if len(self.mOrgTrainData) <= self.MAX_NUM_MEMORY:
            self.readdata_tomemory(kcount, labelcount)
        if len(self.mOrgTrainData) <= 0:
            return None
        orgdata = self.mOrgTrainData[self.mCurIndex]
        start_index = orgdata[0]
        while True:
            if start_index + kcount + labelcount >= len(orgdata[1]):
                self.mOrgTrainData.pop(self.mCurIndex)
                self.mTrainLen -= 1
                orgdata = self.mOrgTrainData[self.mCurIndex]
                start_index = orgdata[0]
                continue
            orgdata = self.mOrgTrainData[self.mCurIndex]
            start_index = orgdata[0]
            break
        self.mCurIndex = (self.mCurIndex+1)%self.mTrainLen
        return orgdata

    def get_train_stock_data_from_database(self, trainCount, kCount, labelCount):
        '''
        获取训练数据
        1、遍历trainCount，每次获取一个训练数据，训练数据返回traindata和labeldata
        2、遍历100个缓存数据，获取kCount天的日线收盘数据，以及labelCount天的涨跌幅作为结果
        :param count:  训练的K线个数，比如100天K线预期后续10天的数据
        :param labelCount: 预期多少天的涨跌幅，比如10天
        :param trainCount: 训练时一次性喂给tensorflow数据量
        :return:
        '''
        if kCount <= 0:
            return None
        train_data = []
        label_data = []

        for index in range(trainCount):
            # orgData is al tuple:(date,open,close,high,low,volume)
            # 一次获取count个股票的
            orgdata = self.getOneOrgStockData(kCount, labelCount)
            k, l = self.__get_final_train_data__(orgdata, kCount, labelCount)
            orgdata[0] += 1
            train_data.append(k)
            label_data.append(l)

        return train_data, label_data

    def __guiyi(self, data_list):
        min_value = min(data_list)
        max_value = max(data_list)
        tempk_list = np.subtract(data_list, min_value)
        try:
            tempk_list = np.multiply(tempk_list, 1.0 / (max_value - min_value))# 归一化操作
        except:
            return None
            pass
        return tempk_list

    def __get_final_train_data__(self, orgdata, kcount, labelcout):
        curindex = orgdata[0]
        datadf = orgdata[1]
        # 收盘价处理
        tempk_list = datadf['close'][curindex:curindex + kcount]
        tempk_list = self.__guiyi(tempk_list)
        # 成交量处理
        tempv_list = datadf['volume'][curindex:curindex + kcount].tolist()
        tempv_list = self.__guiyi(tempv_list)
        # 涨跌处理
        t = datadf['close'][curindex + kcount]
        try:
            tempk_label = (datadf['close'][curindex + kcount + labelcout] - t) / t # 除操作，做下防护
        except:
            tempk_label = 0
            pass
        ret_list = list(tempk_list) + list(tempv_list)
        tempk_label = self.getzd_index(tempk_label)
        return ret_list, tempk_label

    def getzd_index(self, inputzd):
        zd_flags = np.linspace(-0.05, 0.05, 2)
        ret = np.zeros(3)
        i = 0
        for item in zd_flags:
            if inputzd < item:
                break
            i += 1
        ret[i] = 1
        return ret





'''
myKDay = realKDayData_train()
#data = myKDay.get_all_sotck_data_from_database()
train, label =  myKDay.get_train_stock_data_from_database(100, 100, 10)
print(len(train[0]))
del myKDay
'''
#id = '603999'
#ret = myKDay.get_all_stock_data_toDb_byEngine()
#ret = myKDay.get_one_data_form_databases('603999')
#print(ret)
#print(type(ret))
#myKDay.get_one_stock_data_toDb_byEngine(id)