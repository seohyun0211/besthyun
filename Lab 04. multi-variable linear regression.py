x1 = [ 73., 93., 89., 96., 73.]
x2 = [ 80., 88., 91., 98., 66.]
x3 = [ 75., 93., 90., 100., 70.]
#x1,x2,x3은 입력 데이터이다.
Y = [152., 185., 180., 196., 142.]
#Y값은 출력 데이터 즉, 정답이다.

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
#변수가 x1,x2,x3 이렇게 3개 있으므로 초기의 값을 1로 설정한 w값도 3개가 나와야한다.
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001
#learning_rate는 0.00001의 값을 준다.

for i in range(1000+1):
    with tf.GradientTape() as tape:
#tf.GradientTape()은 cost함수의 gradient를 기록하는 함수다.
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
#우리의 가설에서 정답 값인 Y값을 뺀 오차 제곱의 평균값으로 cost를 정의한다.

    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
#w1, w2, w3, b의 기울기는 각각 w1_grad, w2_grad, w3_grad, b_grad이다.

w1.assign_sub(learning_rate * w1_grad)
w2.assign_sub(learning_rate * w2_grad)
w3.assign_sub(learning_rate * w3_grad)
b.assign_sub(learning_rate * b_grad)
#gradient값에 learning_rate값을 곱하고 그 값을 각각 할당해준다.

if i % 50 == 0:
    print("{:5} | {:12.4f}".format(i, cost.numpy()))
#중간 중간 50번 마다 값을 출력하여 확인한다.

<결과 값>
0 | 11325.9121
50 | 135.3618
100 | 11.1817
150 | 9.7940
200 | 9.7687
250 | 9.7587
300 | 9.7489
350 | 9.7389
400 | 9.7292
450 | 9.7194
500 | 9.7096
550 | 9.6999
600 | 9.6903
650 | 9.6806
700 | 9.6709
750 | 9.6612
800 | 9.6517
850 | 9.6421
900 | 9.6325
950 | 9.6229
1000 | 9.6134
#위에 설정한 것 처럼 50번 마다 결과를 한 번씩 출력한다. 

data = np.array([
      x1,  x2,  x3,   y
    [ 73., 80., 75., 152. ],
    [ 93., 88., 93., 185. ],
    [ 89., 91., 90., 180. ],
    [ 96., 98., 100.,196. ],
    [ 73., 66., 70., 142. ]
], dtype=np.float32)
#데이를터를 행렬로 나타낸다.

X = data[:, :-1]
#5행 3열의 matrix이다. 
y = data[:, [-1]]
#y는 마지막 열만 뜻한다.

W = tf.Variable(tf.random_normal([3, 1]))
#변수가 3개(row 수)이고 출력값이 1개이다.
b = tf.Variable(tf.random_normal([1]))

learning_rate=0.00001
#learing_rate는 작은 값으로 지정
def predict(X):
    return tf.matmul(X, W) + b

n_epochs = 2000
for i in range(n_epochs+1):
    with tf.GradientTape() as tape:
##이 것을 2001번 돌려보며 cost를 tape에 저장한다.
        cost = tf.reduce_mean((tf.square(predict(X) - y)))
#예측값 X에서 정답 값인 Y값을 뺀 오차 제곱의 평균값으로 cost지정한다.
    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)    
    b.assign_sub(learning_rate * b_grad)
#gradient값에 learning_rate값을 곱하고 그 값을 각각 할당해준다.
    
    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
#100번마다 중간중간 출력하여 값을 확인한다.

<결과 값>
epoch | cost
0 | 112662.8359
100 | 17.9033
200 | 4.0140
300 | 3.9923
400 | 3.9724
500 | 3.9527
600 | 3.9330
700 | 3.9134
800 | 3.8939
900 | 3.8746
1000 | 3.8553
1100 | 3.8362
1200 | 3.8171
1300 | 3.7981
1400 | 3.7793
1500 | 3.7606
1600 | 3.7419
1700 | 3.7234
1800 | 3.7049
1900 | 3.6866
2000 | 3.6684
#위에 설정한 것처럼 100번에 한 번씩 결과를 출력한다. cost값은 처음에 큰 값에서 다음 값이 급격히 감소하였다.
