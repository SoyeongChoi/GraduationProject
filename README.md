## Minimizing Real-Time Object Detection Network + emotion recognition
### 주제 : 실시간 사물인식 네트워크 경량화 + 감정인식 서비스 구현

#### 목표 :
1. 실시간 사물인식 네트워크(MobileNet + SSD)를 경량화 시킨다.
2. on-Device로 감정인식 서비스를 구현한다.

**프로젝트 설명** :
```
시각장애인을 위해 감정인식 기술을 이용해 상대방의 표정을 인식 한 후 진동을 통해 상대방의 현재 감정을 전달시켜 준다.
현재 감정을 전달할 때 매 순간 전달하는 것이 아닌 감정이 변화하면 변화된 표정을 진동으로 전달한다.
이와 같은 감정인식 기술을 보다 편리하게 이용하기 위해 경량화를 이용해 on-Device에 올릴 수 있도록 한다.
음성이 아닌 진동서비스를 이용한 이유는 시각장애인 중 청각적으로도 불편한 사람도 편히 사용할 수 있게 하기 위해서이다.
```

### 저장소 구조 :
```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
├── model               - this folder contains any model of your project.
│   └── example_model.py
│
├── trainer             - this folder contains trainers of your project.
│   └── example_trainer.py
│
├── mains              - here's the main(s) of your project (you may need more than one main).
│   └── example_main.py  - here's an example of main that is responsible for the whole pipeline.
│
├── data _loader
│   └── data_generator.py  - here's the data_generator that is responsible for all data handling.
│
└── utils
	├── logger.py
	└── any_other_utils_you_need

```


## GIT HUB 사용하기
#### Git은 Develop 에서 진행하고 소스트리를 이용한다.
#### Git Branch : git branch는 feature별로 나누어 진행한다. 상위폴더는 수정코드의 상위폴더이다.
		Develop/상위폴더/구현기능(최대한 간결하게)
		ex) Develop/Model/loss구현
#### Commit : commit은 제목과 내용으로 이루어져있다.
		제목 : [고친파일 명] 구현기능(branch와 같거나 살짝 길게)
		내용 : 자유롭게
		ex) [train]loss 구현
			ssd loss function을 오픈소스를 이용해 구현함
		* Commit 제목과 내용 사이에는 공백 한줄 추가


## Temporary Code 설명 ##
:
create_prior_box.py : ssd 학습을 위한 전처리와, 예측한 후처리에 사용하기 위한를 prior_box를 미리 생성
-> new_ssd_300.pkl : prior_box의 정보가 저장된 파일

create_prior_box2.py : ssd 학습을 위한 전처리와, 예측한 후처리에 사용하기 위한를 prior_box를 미리 생성(aspect ratio를 2.0 => 1.5 / 3.0 => 2.0로 변경)
-> new_ssd2_300.pkl : aspect ratio가 변경된 prior_box의 정보가 저장된 파일

get_data_from_XML.py : face_data에 대한 정보를 xml로 부터 읽어와서 p파일(dictionary 구조)로 저장.
-> face_data.p 

train.py : 모델을 train 하여 graph를 생성하는 코드. (+ model, quantization 포함)
test.py : weight를 불러와 test하는 코드.

SSDLoss_new.py : SSD 학습을 위한 Loss Function

ssd_utils.py : SSD 학습을 위한 y_true 생성(전처리), detection 후 화면에 표시하기 위한 형태의 데이터를 생성(후처리)

generate_tfrecord.py : ssd의 postprocessing을 위해 tfrecord 파일을 생성하는 코드
