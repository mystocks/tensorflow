#coding=utf-8


from TrainKDataSet import realKDayData_train
import tensorflow as tf
# 调用input_data获取数据，如果文件已经下载，放
#mnist = realKDayData.read_data_sets("MNIST_data/", one_hot=True)
myKDay = realKDayData_train()
labellen = 3
with tf.name_scope('test') as scope:
  # 图片数据输入
  x = tf.placeholder("float", [None, 200])

  W = tf.Variable(tf.zeros([200,labellen]))
  b = tf.Variable(tf.zeros([labellen]))

  y = tf.nn.softmax(tf.matmul(x,W) + b)
  y_ = tf.placeholder("float", [None,labellen])
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  tf.summary.scalar('loss', cross_entropy)
  train_step = tf.train.GradientDescentOptimizer(0.002).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

for i in range(20000):
    batch_xs, batch_ys = myKDay.get_train_stock_data_from_database(100, 100, 10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
      rs = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
      writer.add_summary(rs, i)
      print("cur progress is ", i)

#print(sess.run(W), sess.run(b))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
for i in range(200):
    batch_xs, batch_ys = myKDay.get_train_stock_data_from_database(100, 100, 10)
    print ("the accuracy is ", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))