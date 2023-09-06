# 2023년 국립국어원 인공 지능 언어 능력 평가


## 감정 분석 과제 : EA


## 이야기 완성 과제 : SC

### Tesk 설명 : ![https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=102&clCd=ING_TASK&subMenuId=sub01](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=102&clCd=ING_TASK&subMenuId=sub01)
이야기 완성 과제는 제공된 문장들을 논리적으로 연결하는 문장을 생성하는 과제이다. 이 과제를 통해 문장들의 맥락을 파악하고 연결고리를 찾는 과정을 통해 기계의 언어 이해 능력을 향상시키는 데 기여할 수 있으며, 이어지는 문장을 생성하게 함으로써 언어 생성 능력을 측정할 수 있다. 이야기 완성 과제는 인공지능 챗봇, 자동 번역, 문서 요약 등 다양한 분야에서 활용될 수 있다.

<img width="649" alt="Screenshot 2023-09-06 at 4 16 22 PM" src="https://github.com/Docent-Inc/Korean_2023/assets/89565530/7cb7f2e1-6cb7-4994-b7ad-56a8ea3caf07">

### BaseModel : nlpai-lab/kullm-polyglot-5.8b-v2

### 학습 및 튜닝 방법
- 초기 학습: 우선, nlpai-lab/kullm-polyglot-5.8b-v2를 기반으로 하는 BaseModel을 2 epoch 동안 학습시킵니다.
- 개발 세트 평가: 학습된 모델을 개발(dev) 세트에 적용하여 성능을 평가합니다.
- 결과 분석: 원문과 모델의 출력을 비교하여 자연스럽지 않은 결과를 내놓은 데이터셋을 식별합니다.
- 추가 학습: 식별된 문제점을 개선하기 위해 해당 데이터셋을 추가로 학습시킵니다.
- 반복적 학습: 위의 과정을 반복하면서 모델을 튜닝합니다.

### 학습 전략
- LoRA (Low-Rank Adaptation of Large Language Models): LoRA는 PEFT(Parameter Effecient Fine-Tuning)의 기법 중 하나입니다. Pre-trained model의 weight는 고정한 채로, 몇 개의 dense(fc) layer만 학습시켜 downstream task의 연산량을 줄일 수 있습니다.
- RLHF (Reinforcement Learning from Human Feedback): 모델의 출력을 개선하기 위해 인간의 피드백을 이용한 강화학습을 적용했습니다.

### 
