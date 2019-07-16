## Minimizing Real-Time Object Detection Network + emotion recognition
### 주제 : 실시간 사물인식 네트워크 경량화 + 감정인식 서비스 구현

#### 목표 :
1. 실시간 사물인식 네트워크를 경량화 시킨다.
2. on-Device로 감정인식 서비스를 구현한다.

**프로젝트 설명** : 

          시각장애인을 위해 감정인식 기술을 이용해 상대방의 표정을 인식 한 후 진동을 통해 상대방의 현재 감정을 전달시켜 준다. 현재 감정을 전달할 때 매 순간 전달하는 것이 아닌 감정이 변화하면 변화된 표정을 진동으로 전달한다. 이와 같은 감정인식 기술을 보다 편리하게 이용하기 위해 경량화를 이용해 on-Device에 올릴 수 있도록 한다. 음성이 아닌 진동서비스를 이용한 이유는 시각장애인 중 청각적으로도 불편한 사람도 편히 사용할 수 있게 하기 위해서이다.






## GIT HUB 사용하기
#### Git은 Develop 에서 진행하고 소스트리를 이용한다.
#### Git Branch : git branch는 feature별로 나누어 진행한다. 상위폴더는 수정코드의 상위폴더이다. 
              Develop/상위폴더/구현기능(최대한 간결하게)
            ex) Develop/Model/loss구현
#### Commit : commit은 제목과 내용으로 이루어져있다.
              제목 : [고친파일 명] 구현기능(branch와 같거나 살짝 길게)
              내용 : 자유롭게
            ex)  [train]loss 구현
                 ssd loss function을 오픈소스를 이용해 구현함
