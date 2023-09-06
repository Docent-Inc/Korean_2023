# 2023년 국립국어원 인공 지능 언어 능력 평가


## 감정 분석 과제 : EA


## 이야기 완성 과제 : SC
### BaseModel : nlpai-lab/kullm-polyglot-5.8b-v2
### 학습 및 튜닝 방법
- 초기 학습: 우선, nlpai-lab/kullm-polyglot-5.8b-v2를 기반으로 하는 BaseModel을 2 epoch 동안 학습시킵니다.
- 개발 세트 평가: 학습된 모델을 개발(dev) 세트에 적용하여 성능을 평가합니다.
- 결과 분석: 원문과 모델의 출력을 비교하여 자연스럽지 않은 결과를 내놓은 데이터셋을 식별합니다.
- 추가 학습: 식별된 문제점을 개선하기 위해 해당 데이터셋을 추가로 학습시킵니다.
- 반복적 학습: 위의 과정을 반복하면서 모델을 튜닝합니다.
### 학습 전략
![image](https://github.com/Docent-Inc/Korean_2023/assets/89565530/bcaee0b4-29f1-413a-aa6d-ae299845630d)
- LoRA (Low-Rank Adaptation of Large Language Models): LoRA는 PEFT(Parameter Effecient Fine-Tuning)의 기법 중 하나입니다. Pre-trained model의 weight는 고정한 채로, 몇 개의 dense(fc) layer만 학습시켜 downstream task의 연산량을 줄일 수 있습니다.
- RLHF (Reinforcement Learning from Human Feedback): 모델의 출력을 개선하기 위해 인간의 피드백을 이용한 강화학습을 적용했습니다.
