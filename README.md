# ♥WELCOME TO SEOHYUN'S GITHUB♥
* Let's start to review tensorflow's code
 
[Lab02. simple linear regression 코드 리뷰 바로가기 click!](https://github.com/seohyun0211/besthyun/blob/master/Lab2.%20simple%20linear%20regression.py)

 
 
 

* import tensorflow as tf  
  #tensorflow 라이브러리를 tf라는 약어로 불러오겠다.
* import numpy as np
  #numpy 라이브러리를 np라는 약어로 불러오겠다.
* tf.enable_eager_execution()  
  #즉시 실행을 시작한다.

* x_data = [1, 2, 3, 4, 5]
  #x데이터의 값은 1,2,3,4,5 이다.
* y_data = [1, 2, 3, 4, 5]
  #y데이터의 값은 1,2,3,4,5d이다.

* import matplotlib.pyplot as plt
  #matplotlib 라이브러리를 plt라는 약어로 불러오겠다.
* plt.plot(x_data, y_data, 'o')
  #
* plt.ylim(0, 8)
  #


x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]
W = tf.Variable(2.9)
b = tf.Variable(0.5)
# hypothesis = W * x + b
hypothesis = W * x_data + b











# Lab 3. 











# Lab 4. multi-variable linear regression

* import tensorflow as tf
  #tensorflow 라이브러리를 tf라는 약어로 불러온다.
* tf.set_random_seed(777)
  #난수 777seed를 제공한다.
  
* filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')
  #string_input_producer 메소드를 사용해 여러개의 파일을 입력하고 shuffle여부와 queue의 이름을 지정한다.
  
* reader = tf.TextLineReader() 
  #텍스트파일에서 한 줄씩 읽어라
* key, value = reader.read(filename_queue)
  #textlinereader를 사용해 key와 value를 가져온다.



