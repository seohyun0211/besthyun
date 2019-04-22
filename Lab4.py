import tensorflow as tf
#tensorflow 라이브러리를 tf라는 약어로 불러오겠다.
tf.set_random_seed(777) 
#777개의 난수를 임의로 제공한다.

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')
#string_input_producer 메소드를 사용해 여러개의 파일을 입력하고 shuffle여부와 queue의 이름을 지정한다.

reader = tf.TextLineReader()
#텍스트파일에서 한 줄씩 읽어라.
key, value = reader.read(filename_queue)
#textlinereader를 사용해 key와 value를 가져온다.

record_defaults = [[0.], [0.], [0.], [0.]]
#record_defaults는 해당 필드에 데이터가 없는 경우, 기본값을 채워주기 위해 사용한다.
xy = tf.decode_csv(value, record_defaults=record_defaults)
#불러온 파일을 decode처리한다. value형태는 각각의 필드로 구분된 상태가 아닌 하나의 문자로 인식한다. 그래서 이를 각각의 필드로 구분해주기 위한 decode하는 절차가 필요하다.

train_x_batch, train_y_batch = /
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 3])
#데이터의 shape을 지정해준다. 데이터 셋이 얼만큼 있을지 모르기 때문에 행 부분은 None으로 처리한다.
Y = tf.placeholder(tf.float32, shape=[None, 1])
#데이터의 shape을 지정해준다. 데이터 셋이 얼만큼 있을지 모르기 때문에 행 부분은 None으로 처리한다.

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
#변수의 값을 정할 때 variable()함수를 사용한다. W값의 이름은 weight이다.
b = tf.Variable(tf.random_normal([1]), name='bias')
#변수의 값을 정할 때 variable()함수를 사용한다. b값의 이름은 bias이다.

hypothesis = tf.matmul(X, W) + b
#hypothesis를 행렬로 나타낸다.

cost = tf.reduce_mean(tf.square(hypothesis - Y))
#가설과 실제값의 평균값 차이를 최소화한다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
#cost를 최소화할 수 있는 함수를 마련한다.

sess = tf.Session()
#세션을 생성하는 과정을 통틀어 sess라고 부르겠다.
sess.run(tf.global_variables_initializer())
#오류를 방지하기 위해 변수는 세션을 만든 후 꼭 처음으로 초기화를 해준다.

coord = tf.train.Coordinator()
#coordinator을 생성한다.
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#그래프에 추가된 queue runners를 threads로 실행한다.

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
#2000번 루프 돌면서 학습을 시킨다, feed_dict 에 학습된 결과가 저장된다. 10번에 한 번씩 결과값을 출력하여 확인한다.


print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
#학습한 내용을 가지고 결과값을 출력한다.
print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
#학습한 내용을 가지고 결과값을 출력한다.
