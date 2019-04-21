# ♥WELCOME TO SEOHYUN'S GITHUB♥
* Let's start to review tensorflow's code

 ![2 1](https://user-images.githubusercontent.com/49617386/56468886-57471580-646d-11e9-81a6-bc471189ba14.png)
 
[Lab02. simple linear regression 코드 리뷰 바로가기 click!](https://github.com/seohyun0211/besthyun/blob/master/Lab2.%20simple%20linear%20regression.py)

 ![3 1](https://user-images.githubusercontent.com/49617386/56468967-611d4880-646e-11e9-8cbb-3293cd2122f2.png)

[Lab03. linear regression How to minimize cost LAB 코드 리뷰 바로가기 click!]
()









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



