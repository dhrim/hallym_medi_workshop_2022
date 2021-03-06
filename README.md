# 강남성심병원 워크샾 2022

<br>

# 목표

의학 연구자 혹은 임상의가 딥러닝을 파악하여 실제 연구에 활용할 수 있는 기반을 마련한다.

## 상세 목표

- 딥러닝 개념 파악
- 딥러닝 학습의 실체 파악
- 데이터 준비와 모양 파악
- 딥러닝 환경과 코드, 실행방법 파악
- 딥러닝 작업의 종류 파악

<br>

- 딥러닝을 사용한 논문 유형 파악
- 연구를 위한 딥러닝 적용 대상 파악
- 연구를 위한 딥러닝 데이터 파악
- 연구를 위한 딥러닝 실험방법 파악

<br>

# 진행 일정

2022년 7월 23일 ~ 7월 24일

## 1일차
목표 : 
  딥러닝을 파악한다.
  딥러닝 동작을 실제 코드로 파악한다.

- 09:00 ~ 09:50 딥러닝의 학습 이해 #1
- 10:00 ~ 10:50 딥러닝의 학습 이해 #2
- 11:30 ~ 11:50 DNN의 함수근사화 능력, 인공지능/머신러닝/딥러닝 개념, 딥러닝 상세
- 13:00 ~ 13:50 Keras를 사용한 DNN #1
- 14:00 ~ 14:50 Keras를 사용한 DNN #2
- 15:00 ~ 15:50 DNN - IRIS 분류
- 16:00 ~ 16:50 DNN - MNIST 분류
- 17:00 ~ 17:50 CNN - MNIST 분류, CIFAR10 분류

<br>

## 2일차
목표 : 실제 사용되는 딥러닝 모델과 논문에서의 딥러닝 사용을 파악하고, 새로운 연구를 기획한다.

- 09:00 ~ 09:50 AutoEncoder와 활용
- 10:00 ~ 10:50 비지도 학습, 강화 학습, 알파고 이해하기
- 11:00 ~ 11:50 VGG16을 사용한 영상 분류, 커스텀 데이터 학습
- 13:00 ~ 13:50 U-Net을 사용한 영상 영역 분할
- 14:00 ~ 14:50 딥러닝을 사용한 논문 리뷰 #1
- 15:00 ~ 15:50 딥러닝을 사용한 논문 리뷰 #2
- 16:00 ~ 16:50 딥러닝 논문 실험 재현 #1
- 17:00 ~ 17:50 딥러닝 논문 실험 재현 #2


<br>

# 교육 자료


- 딥러닝 개요
    - 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)
    - 알파고 이해하기 : [understanding_alphago.pptx](material/deep_learning/understanding_alphago.pptx)
- Keras
    - DNN in Keras : [dnn_in_keras.ipynb](./material/deep_learning/dnn_in_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/dnn_in_keras.ipynb)
    - Keras 요약 [keras_in_short.md](material/deep_learning/keras_in_short.md)
- DNN as classifier
    - 속성 데이터 IRIS 분류 : [dnn_iris_classification.ipynb](./material/deep_learning/dnn_iris_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/dnn_iris_classification.ipynb)
    - 흑백 영상 데이터 MNIST 분류 : [dnn_mnist.ipynb](./material/deep_learning/dnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/dnn_mnist.ipynb)

- CNN
    - CNN의 이해 : [deep_learning_intro.pptx](./material/deep_learning/deep_learning_intro.pptx)
    - 흑백 영상 데이터 MNIST 영상분류 : [cnn_mnist.ipynb](./material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/cnn_mnist.ipynb)
    - CIFAR10 컬러영상분류 : [cnn_cifar10.ipynb](./material/deep_learning/cnn_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/cnn_cifar10.ipynb)

- VGG16 전이학습 : [VGG16_classification_and_cumtom_data_training.ipynb](./material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)

- U-Net Segmentation - Lung data : [unet_segementation.ipynb](./material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/unet_segementation.ipynb)

- AutoEncoder
    - AutoEncoder 실습 : [autoencoder.ipynb](./material/deep_learning/autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/autoencoder.ipynb)
    - 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](./material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/denoising_autoencoder.ipynb)
    - Super Resolution : [mnist_super_resolution.ipynb](./material/deep_learning/mnist_super_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/mnist_super_resolution.ipynb)
<br>

- [딥러닝 논문 리뷰](https://docs.google.com/presentation/d/1SZ-m4XVepS94jzXDL8VFMN2dh9s6jaN5fVsNhQ1qwEU/edit?usp=sharing)


<br>

# 기타 자료

- [흥미로운 딥러닝 결과](material/deep_learning/some_interesting_deep_learning.pptx)
- [yolo를 사용한 실시간 불량품 탐지 사례](https://drive.google.com/file/d/194UpsjG7MyEvWlmJeqfcocD-h-zy_4mR/view?usp=sharing)
- [GAN을 사용한 생산설비 이상 탐지](material/deep_learning/anomaly_detection_using_gan.pptx)
- [딥러닝 이상탐지](./material/deep_learning/deep_learning_anomaly_detection.pptx)
- [이상탐지 동영상](material/deep_learning/drillai_anomaly_detect.mp4)
- Object Detection : [keras_yolov3.ipynb](keras_yolov3.ipynb)   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/keras_yolov3.ipynb)
- RNN을 사용한 영화 평가 데이터 IMDB 분류 : [rnn_text_classification.ipynb](./material/deep_learning/rnn_text_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/rnn_text_classification.ipynb)
- ROC, AUC : [roc_auc_confusion_matric.ipynb](./material/deep_learning/roc_auc_confusion_matric.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/roc_auc_confusion_matric.ipynb)


<br>


# Template

- 속성 데이터
    - 예측 : [template_attribute_data_regression.ipynb](material/deep_learning/template_attribute_data_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_attribute_data_regression.ipynb)
    - 분류 : [template_attribute_data_classification.ipynb](material/deep_learning/template_attribute_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_attribute_data_classification.ipynb)
    - 2진 분류 : [template_attribute_data_binary_classification.ipynb](material/deep_learning/template_attribute_data_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_attribute_data_binary_classification.ipynb)    
- 영상 데이터
    - 예측 - vanilla CNN : [template_image_data_vanilla_cnn_regression.ipynb](material/deep_learning/template_image_data_vanilla_cnn_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_image_data_vanilla_cnn_regression.ipynb)
    - 예측 - 전이학습 : [template_image_data_transfer_learning_regression.ipynb](material/deep_learning/template_image_data_transfer_learning_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_image_data_transfer_learning_regression.ipynb)
    - 분류 - vanilla CNN : [template_image_data_vanilla_cnn_classification.ipynb](material/deep_learning/template_image_data_vanilla_cnn_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_image_data_vanilla_cnn_classification.ipynb)
    - 분류 - 전이학습 : [template_image_data_transfer_learning_classification.ipynb](material/deep_learning/template_image_data_transfer_learning_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_image_data_transfer_learning_classification.ipynb)
    - 2진 분류 - vanilla CNN : [template_image_data_vanilla_cnn_binary_classification.ipynb](material/deep_learning/template_image_data_vanilla_cnn_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_image_data_vanilla_cnn_binary_classification.ipynb)
    - 2진 분류 - 전이학습 : [template_image_data_transfer_learning_binary_classification.ipynb](material/deep_learning/template_image_data_transfer_learning_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_image_data_transfer_learning_binary_classification.ipynb)
- 순차열 데이터
    - 숫자열
        - 단일 숫자열 예측 : [template_numeric_sequence_data_prediction.ipynb](material/deep_learning/template_numeric_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_numeric_sequence_data_prediction.ipynb)
        - 단일 숫자열 분류 : [template_numeric_sequence_data_classification.ipynb](material/deep_learning/template_numeric_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_numeric_sequence_data_classification.ipynb)
        - 다중 숫자열 분류 : [template_multi_numeric_sequence_data_classification.ipynb](material/deep_learning/template_multi_numeric_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_multi_numeric_sequence_data_classification.ipynb) 
        - 다중 숫자열 다중 예측 : [template_multi_numeric_sequence_data_multi_prediction.ipynb](material/deep_learning/template_multi_numeric_sequence_data_multi_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_multi_numeric_sequence_data_multi_prediction.ipynb)
        - 다중 숫자열 단일 예측 : [template_multi_numeric_sequence_data_one_prediction.ipynb](material/deep_learning/template_multi_numeric_sequence_data_one_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_multi_numeric_sequence_data_one_prediction.ipynb)
    - 문자열
        - 문자열 예측 : [template_text_sequence_data_prediction.ipynb](material/deep_learning/template_text_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_text_sequence_data_prediction.ipynb)
        - 문자열 분류 : [template_text_sequence_data_classification.ipynb](material/deep_learning/template_text_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_text_sequence_data_classification.ipynb)
        - 문자열 연속 예측 : [template_text_data_sequential_generation.ipynb](material/deep_learning/template_text_data_sequential_generation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_text_data_sequential_generation.ipynb)
    - 단어열
        - 단어열 분류 : [template_word_sequence_data_classification.ipynb](material/deep_learning/template_word_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_word_sequence_data_classification.ipynb)
        - 단어열 예측 : [template_word_sequence_data_prediction.ipynb](material/deep_learning/template_word_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_word_sequence_data_prediction.ipynb)
        - 한글 단어열 분류 : [template_korean_word_sequence_data_classification.ipynb](material/deep_learning/template_korean_word_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/template_korean_word_sequence_data_classification.ipynb)
        - Bert를 사용한 한글 문장 간 관계 분류 : [korean_sentence_relation_classification_with_bert.ipynb](./material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb)
        - Bert를 사용한 한글 문장 간 관계값 예측 : [korean_sentence_relation_regression_with_bert.ipynb](./material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/hallym_medi_workshop_2022/blob/main/material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb)


<br>


# 강사

임도형, dh-rim@hanmail.net
