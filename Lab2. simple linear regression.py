import tensorflow as tf
#tensorflow 라이브러리를 tf라는 약어로 불러오겠다.
tf.enable_eager_excution()
#즉시 실행을 시작한다.

x_data = [1,2,3,4,5]
#x데이터 값을 입력시킨다.
y_data = [1,2,3,4,5]
#y데이터 값을 입력시킨다.

w = tf.Variable(2.9)
#w에 임의의 값 2.9를 지정한다.
b = tf.Variable(0.5)
#b에 임의의 값 0.5를 지정한다.

learning_rate = 0.01
#학습률의 값을 0.01로 지정한다. 보통 0.01~0.001이 적당한 학습률이라고 한다.

for i in range(100+1):
    #학습을 101번 반복한다.
    with tf.GradientTape() as tape:
        #안에서 계산을 하면 tape에 계산 과정을 기록해둔다. 후에 tape.gradient를 이용해서 미분을 자동으로 구할 수 있다.
        hypothesis = w * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        #기울기와 y절편에 대한 적합성을 판단하는 중요한 코드이다.
    w_grad, b_grad = tape.gradient(cost, [w,b])
    #위에 기록해둔 tape에 있는 계산 과정들을 gradient함수를 이용하여 cost함수에 대한 w와 b값을 각각 w_grad, b_grad이라고 지정한다.
    w.assign_sub(learning_rate * w_grad)
    #w값을 업데이트 한다. 텐서플로우에서 a.assign_sub(b)는 일반적인 파이썬 코드에서 a = a - b와 같다.
    b.assign_sub(learning_rate * b_grad)
    #b값을 업데이트 한다.
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))
        #w와 b의 값을 확인하기 위해 10번에 한 번씩 값을 출력한다.

        <결과 값>
WARNING:tensorflow:From C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:642: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
    0|    2.4520|     0.376| 45.660004
   10|    1.1036|  0.003398|  0.206336
   20|    1.0128|  -0.02091|  0.001026
   30|    1.0065|  -0.02184|  0.000093
   40|    1.0059|  -0.02123|  0.000083
   50|    1.0057|  -0.02053|  0.000077
   60|    1.0055|  -0.01984|  0.000072
   70|    1.0053|  -0.01918|  0.000067
   80|    1.0051|  -0.01854|  0.000063
   90|    1.0050|  -0.01793|  0.000059
  100|    1.0048|  -0.01733|  0.000055
  # i       w        b          cost
  #위에 설정한 것 처럼 i값이 10번에 한 번씩 출력된다.
  #w값은 점점 1에 가까워지고 b값은 점점 0에 가까워진다.
  #우리가 설정한 값과 cost값의 차이를 줄여햐한다. 
