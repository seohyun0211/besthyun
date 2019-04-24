x1 = [ 73., 93., 89., 96., 73.]
x2 = [ 80., 88., 91., 98., 66.]
x3 = [ 75., 93., 90., 100., 70.]
#x1,x2,x3은 입 데이터이다.
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

data = np.array([
      x1,  x2,  x3,   y
    [ 73., 80., 75., 152. ],
    [ 93., 88., 93., 185. ],
    [ 89., 91., 90., 180. ],
    [ 96., 98., 100.,196. ],
    [ 73., 66., 70., 142. ]
], dtype=np.float32)

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






#학습한 내용을 가지고 결과값을 출력한다.
