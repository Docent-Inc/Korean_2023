# 2023년 국립국어원 인공 지능 언어 능력 평가

<p align="center"><img src="https://github.com/Docent-Inc/Korean_2023/assets/89565530/196e185e-c5cf-446e-ab41-176bf0f60158"></p>

## 감정 분석 과제 : EA

<p align="center"><img width="657" alt="Screenshot 2023-08-30 at 4 06 05 PM" src="https://github.com/Docent-Inc/Korean_2023/assets/89565530/d4e4e8ed-276e-42ca-bc6a-39cf3db43fe5"></p>

### Task 설명
감정 분석은 주어진 텍스트에 대한 화자의 감정 상태를 파악하는 과제이다. 이 과제는 텍스트에 드러나는 8가지 감정 유형을 분류하는 것을 목표로 한다. 감정 분석은 고객 서비스, 사회 네트워크 분석, 피드백 시스템, 인공지능 대화 시스템 등에 널리 활용된다. 외부 데이터 추가 사용 불가, 외부 API 이용 불가, RTX 4090 24GB 1개에서 구동 가능한 모델만 사용 가능

### BaseModel
- beomi/KcELECTRA-base-v2022

### 학습 전략
- StratifiedKFold를 사용해 9개의 fold로 나누어 학습합니다. 
- early stopping을 사용해 학습을 조기 종료하고, 가장 좋은 성능을 보인 모델을 저장합니다.
- 학습률과 배치사이즈를 조절하며 최적의 loss를 만들 수 있도록 합니다.

### 추론 전략
- 9개의 fold로 나누어 학습한 모델을 사용해 추론합니다.
- 각 fold의 추론 결과를 앙상블하여 최종 결과를 도출합니다.

## 이야기 완성 과제 : SC

<p align="center"><img src="https://github.com/Docent-Inc/Korean_2023/assets/89565530/45cc215e-d7cd-4184-8e9b-7503c3433450"></p>

### Tesk 설명
이야기 완성 과제는 제공된 문장들을 논리적으로 연결하는 문장을 생성하는 과제이다. 이 과제를 통해 문장들의 맥락을 파악하고 연결고리를 찾는 과정을 통해 기계의 언어 이해 능력을 향상시키는 데 기여할 수 있으며, 이어지는 문장을 생성하게 함으로써 언어 생성 능력을 측정할 수 있다. 이야기 완성 과제는 인공지능 챗봇, 자동 번역, 문서 요약 등 다양한 분야에서 활용될 수 있다. 외부 데이터 추가 사용 불가, 외부 API 이용 불가, RTX 4090 24GB 1개에서 구동 가능한 모델만 사용 가능

### BaseModel
- nlpai-lab/kullm-polyglot-5.8b-v2

### 학습 전략
- LoRA (Low-Rank Adaptation of Large Language Models): LoRA는 PEFT(Parameter Effecient Fine-Tuning)의 기법 중 하나입니다. Pre-trained model의 weight는 고정한 채로, 몇 개의 dense(fc) layer만 학습시켜 downstream task의 연산량을 줄일 수 있습니다.
- 8비트 양자화를 통해 한정된 GPU 메모리에서도 모델을 학습할 수 있도록 합니다.
- 학습률과 배치사이즈를 조절하며 최적의 loss를 만들 수 있도록 합니다.

### 추론 전략
- beam search의 계수를 최대한 늘려 더 정확한 문장을 생성하도록 유도합니다.
- 학습과 같은 prompt를 사용해 학습효과를 최대한 사용하도록 유도합니다.

### 이중 모델 학습 및 예측 적용

<p align="center"><img width="735" alt="Screenshot 2024-01-25 at 5 47 11 PM" src="https://github.com/Docent-Inc/Korean_2023/assets/89565530/9cb74e25-2945-4ba8-a599-43a3a691dfbb"></p>

- 모델 A: s1, s3을 사용하여 s2를 예측, 모델 B: s1, s2를 바탕으로 s3 예측.
- 모델 A, B를 각각 학습시킨 후, 모델 A를 통해 s2를 예측하고, 모델 B를 통해 s3를 예측합니다.
- 실제 s3와의 차이를 reward로 사용하여 모델 A를 학습시킵니다.
