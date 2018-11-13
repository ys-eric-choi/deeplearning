## ANN Test

#### Summary
- 당뇨병 환자 데이터를 학습하여 당뇨병 여부를 분류하는 Classifier를 ANN으로 구현
- Hidden layer가 없는 model 에서는 77% 정도의 성능을 보임
- 선형 분류로 풀 수 없는 문제를 hidden layer를 이용하여 성능을 높이는 것을 목표로 함
- Hidden layer를 쌓는 것을 Tensorflow와 Keras로 각각 구현하는 실습 진행

#### TODO
- Hidden layer의 수와 같은 network 구조 보다는 optimizer에 따라 성능이 크게 차이나 이를 확인할 필요가 있음
- Keras에서는 ReLU 함수를 사용하여 성능을 높였으나 Tensorflow에서는 ReLU 함수를 사용하면 학습이 되지 않는 이유
- Tensorflow에서 weight 수렴 후 epochs를 계속 돌리면 cost value가 'nan'으로 출력되며 학습이 되지 않으면서 학습 결과가 날아가는 이유

<br><br>

#### 2018.11.13 수정사항
- tensorflow를 이용한 ann 구현 시 학습 중 NaN이 출력되며 학습이 되지 않는 현상과
  ReLU 함수를 사용할 수 없는 현상에 대해 아래 블로그에서 원인을 찾을 수 있었다.

> URL: <http://blog.naver.com/gyrbsdl18/221068979134>
```
결론 부터 말하자면,

가급적이면 수식을 직접 구현하지 말고, [tf.losses, tf.contrib.losses, tf.nn] 등에 미리 구현된 함수를 사용해야 한다.

그 이유는, exp(x) 함수의 값이 지수적으로 증가하므로,

x가 어느 정도만 ( e.g 800 ) 커져도 overflow를 일으키기 때문이다.
[출처] cross entropy loss function 이 nan이나 inf 결과를 내는 이유|작성자 박효균
```

- 코드를 아래와 같이 수정 후 epochs이 크거나 ReLU함수를 사용해도 학습이 정상적으로 이뤄지는 것을 확인
> ann_tensor_1.py
```python
 54 #model = tf.sigmoid(tf.matmul(H5, W_o) + b_o)
 55 model  = tf.matmul(H5, W_o) + b_o
 56 #cost = tf.reduce_mean(-Y * tf.log(model) - (1 - Y) * tf.log(1 - model))
 57 # cost 계산 수식을 이미 만들어진 함수를 사용
 58 cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=Y)
```
