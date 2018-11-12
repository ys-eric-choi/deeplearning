## Deep Learning

Deep Learning을 공부하면서 배운 내용들을 정리
 - DNN, RNN, CNN, LSTM, etc.
  
  
 
 
---

## anaconda를 이용한 tensorflow 설치 시 주의사항

복습을 위하여 macbook에 아나콘다를 설치하고 아래와 같이 가상환경 구축 후 tensorflow를 설치를 시도하였다.

```
$ conda create -n "tensorflow" python=3
$ source activate tensorflow
$ pip install tensorfow
```

그러나 아래와 같은 message가 출력되면서 설치가 되지 않았다...
```
$ pip install tensorflow
Collecting tensorflow
Could not find a version that satisfies the requirement tensorflow (from versions: )
No matching distribution found for tensorflow
```

여러가지 원인을 찾아본 결과 하나의 블로그에서 나의 실수를 찾을 수 있었다!

> 출처: http://disq.us/p/1wflqob 페이지의 Hyeonwook Kim 님 댓글

> 현재 기준)python 최신버전을 설치하실 경우, 3.7버전이 설치됩니다. (tensorflow 설치페이지에는 3.6버전 까지만 지원한다고 되어있습니다.)
python 3.6버전을 설치하셔도 되고, 저는 https://stackoverflow.com/questions/38896424/tensorflow-not-found-using-pip/42596864#42596864 를 참고하셔서 해결하였습니다.

결론은 내 macbook에는 python 3.7 버전이 설치되어 있었기 때문에  
anaconda 가상환경 구축 시 python=3 옵션을 주면 3.7버전을 사용하게 된다.

따라서 아래와 같이 pyhton 버전을 명시하여 가상환경을 구축하여
정상적으로 tensorflow를 설치할 수 있었다.

```
$ conda create -n "tensorflow" python=3.5
$ source activate tensorflow
$ pip install tensorflow
```

역시 기초 없이 주먹구구 식으로 하다보니 이런 일이 자주 생기는 것 같다...
