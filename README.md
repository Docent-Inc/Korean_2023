# 2023년 국립국어원 인공 지능 언어 능력 평가


## 감정 분석 과제 : EA


## 이야기 완성 과제 : SC

### Tesk 설명
이야기 완성 과제는 제공된 문장들을 논리적으로 연결하는 문장을 생성하는 과제이다. 이 과제를 통해 문장들의 맥락을 파악하고 연결고리를 찾는 과정을 통해 기계의 언어 이해 능력을 향상시키는 데 기여할 수 있으며, 이어지는 문장을 생성하게 함으로써 언어 생성 능력을 측정할 수 있다. 이야기 완성 과제는 인공지능 챗봇, 자동 번역, 문서 요약 등 다양한 분야에서 활용될 수 있다.

### BaseModel
- nlpai-lab/kullm-polyglot-5.8b-v2

### 학습 방법
- nlpai-lab/kullm-polyglot-5.8b-v2인 BaseModel을 4 batch_size로 2 epoch 동안 학습시킵니다.
- "문맥과 문법적 정확성 및 논리적 일관성에 맞는 자연스러운 한 문장이 되도록 두 문장 사이에 들어갈 한 문장을 접속사를 신경써서 만들어주세요."라는 프롬프트와 함께 학습시킵니다.

### 학습 전략
- LoRA (Low-Rank Adaptation of Large Language Models): LoRA는 PEFT(Parameter Effecient Fine-Tuning)의 기법 중 하나입니다. Pre-trained model의 weight는 고정한 채로, 몇 개의 dense(fc) layer만 학습시켜 downstream task의 연산량을 줄일 수 있습니다.
- 학습률과 배치사이즈를 조절하며 최적의 loss를 만들 수 있도록 합니다.

### 추론 전략
- beam search의 계수를 최대한 늘려 더 정확한 문장을 생성하도록 유도합니다.
- 학습과 같은 prompt를 사용해 학습효과를 최대한 사용하도록 유도합니다.

