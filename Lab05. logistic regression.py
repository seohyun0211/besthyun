import tensorflow.contrib.eager as tfe
#tensorflow에서 eager모드로 실행하기 위한 라이브러리를 실행한다.
tf.enable_eager_execution()
#즉시 실행을 시작한다.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
#tf.data를 통해 원하는 x값과 y값을 실제 x의 길이 만큼 뱃치로 학습을 하겠다.
W = tf.Variable(tf.zeros([2,1]), name='weight')
#W의 값을 정의하고 이름은 weight으로 한다. tf.zero([2,1])즉 2행 1열이다.
b = tf.Variable(tf.zeros([1]), name='bias')
#b의 값을 정의하고 이름은 bias로 한다.
def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis
#sigmoid함수를 활용하여 가설을 설정한다.
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.log(loss_fn(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
    return cost
#labels값과  hypothesis를 통해 원하는 cost값을 구한다.
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(hypothesis,labels)
    return tape.gradient(loss_value, [W,b])
#가설을 통해 나온 값과 실제 값을 loss를 통해 구한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#GradientDescentOptimizer을 통해 cost값을 줄일 준비를 한다.
for step in range(EPOCHS):
#함수를 EPOCHS만큼 돌린다.
    for features, labels in tfe.Iterator(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
#실제 x값과 y값을 넣어가며 모델이 만들어진다. optimizer를 통해 실제 값과 가장 가까워지도록 계속 minimize한다. 이 과정을 통해 W와 b가 계속 업데이트 된다.  
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features) ,labels)))
#100번 마다 한 번씩 결과를 출력하여 확인한다. 
def accuracy_fn(hypothesis, labels):
#가설과 함수를 비교한다.
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#hypothesis>0.5를 기준으로 정확도를 측정한다. tf.cast()는 true나 false값을 1과 0으로 반환한다.
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels) , dtype=tf.int32))
#실제 값과 예측되어 나온 값이 맞는지를 accuracy를 통해 확인한다.
    return accuracy
    test_acc = accuracy_fn(logistic_regression(x_test),y_test)
#값이 정확한지를 출력하여 모델을 검증한다.
