## ANN Test

#### Summary
- 당뇨병 환자 데이터를 학습하여 당뇨병 여부를 분류하는 Classifier를 ANN으로 구현
- Hidden layer가 없는 model 에서는 77% 정도의 성능을 보임
- 선형 분류로 풀 수 없는 문제를 hidden layer를 이용하여 성능을 높이는 것을 목표로 함
- Hidden layer를 쌓는 것을 Tensorflow와 Keras로 각각 구현하는 실습 진행

#### TODO
- Hidden layer의 개수와 같은 network 구조 보다는 optimizer에 따라 성능이 크게 차이나 이를 확인할 필요가 있음
- Keras에서는 ReLU 함수를 사용하여 성능을 높였으나 Tensorflow에서는 ReLU 함수를 사용하면 학습이 되지 않는 이유
- Tensorflow에서 weight 수렴 후 epochs를 계속 돌리면 cost value가 'nan'으로 출력되며 학습이 되지 않으면서 학습 결과가 날아가는 이유
