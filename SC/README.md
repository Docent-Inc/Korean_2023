# 2023 국립국어원 인공 지능 언어 능력 평가: 이야기 완성 과제

## 프로젝트 소개

이 프로젝트는 2023년 국립국어원에서 주최하는 인공 지능 언어 능력 평가의 이야기 완성 과제를 위한 코드입니다. 본 프로젝트는 SC(Story Completion) 모델을 훈련하고 추론하는 과정을 포함하며, Adapter 모델을 사용하여 훈련된 모델을 효율적으로 활용합니다.

## 설치 및 환경 설정

1. **데이터 셋 준비**: 프로젝트의 경로에 맞게 데이터 셋을 위치시킵니다.
  
2. **환경 설정**: `Korean_2023/SC` 디렉토리로 이동합니다. 그 후, `conda`를 사용하여 새로운 환경을 생성하고 활성화합니다.

    ```bash
    conda create --name sc python=3.8
    conda activate sc
    ```
3. **라이브러리 설치**: 이후 필요한 라이브러리를 설치해줍니다.

    ```bash
    pip install -r requirements.text
    ```

## 모델 훈련

1. 환경이 활성화되면, `train.py` 스크립트를 실행하여 모델을 훈련시킵니다.

    ```bash
    python train.py
    ```

   이 스크립트는 지정된 설정에 따라 모델을 훈련하고 체크포인트를 저장합니다.

## Adapter 모델 병합

1. 훈련이 성공적으로 완료된 후, `src/merge_model.py` 스크립트를 수정하여 Adapter 모델의 경로를 지정합니다.

2. 스크립트를 실행하여 Adapter 모델을 병합합니다.

    ```bash
    python src/merge_model.py
    ```

## 추론 실행

1. 병합된 모델을 사용하여 추론을 실행합니다.

    ```bash
    python inference.py
    ```

이제 추론 결과를 확인할 수 있습니다.
