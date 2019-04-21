import numpy as np
#numpy 라이브러리를 np라는 약어로 불러오겠다.
X = np.array([1, 2, 3])
#numpy에서 배열은 동일한 타입의 값들을 가지며, 배열의 차원을 rank라고 한다. X에 rank가 1인 배열의 데이터를 지정한다.
Y = np.array([1, 2, 3])
#Y에 랭크가 1인 배열의 데이터를 지정한다.
def cost_func(W, X, Y):
#cost함수를 정의한다.
hypothesis = X * W
#가설 설정한다.
return tf.reduce_mean(tf.square(hypothesis - Y))
#hypothesis에서 Y값을 뺀 값을 각각 제곱하고 합계에 대한 평균을 계산한다.
W_values = np.linspace(-3, 5, num=15)
#np.linspace를 이용해서 -3에서 5까지의 구간을 15개로 쪼개서 list로 보여주겠다. 
cost_values = []
for feed_W in W_values:
curr_cost = cost_func(feed_W, X, Y)
cost_values.append(curr_cost)
#list값을 하나하나 선별하여 W값으로 사용을 한다. 그리고 cost가 W값에 따라 어떻게 변하는지 기록한다.
print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
#위에 기록한 값들을 출력한다.
tf.set_random_seed(0)
#random seed를 초기화한다.
x_data = [1., 2., 3., 4.]
#x data를 지정한다.
y_data = [1., 3., 5., 7.]
#y data를 지정한다.
W = tf.Variable(tf.random_normal([1], -100., 100.))
#정규분포를 따르는 random number를 1개짜리로 변수를 만들어서 W에 할당하여 정의한다.
for step in range(300):
#아래 의 것을 300번 반복한다.
hypothesis = W * X
#가설 설정한다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#cost값은 #hypothesis에서 Y값을 뺀 값을 각각 제곱하고 합계에 대한 평균을 계산한 값이다.
alpha = 0.01
gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
#gradient값은 W와 X값의 곱에서 Y,X의 값을 뺀 값들을 곱하고 그 합계에 대한 평균을 계산한 값이다. 
descent = W - tf.multiply(alpha, gradient)
#descent값은 alpha와 gradient값을 곱한 값을 W에서 뺀 값이다.
W.assign(descent)
#W에 descent의 값으로 할당한다.
if step % 10 == 0:
print('{:5} | {:10.4f} | {:10.6f}'.format(
    step, cost.numpy(), W.numpy()[0]))
#값을 확인하기 위해 cost값과 W값을  10번에 한 번씩 값을 출력한다.
