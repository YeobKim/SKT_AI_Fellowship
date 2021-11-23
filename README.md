# SKT_AI_Fellowship

안녕하세요, SKT AI Fellowship 3기 Team SKSAK의 김재엽입니다.

이 레퍼지토리는 [SKT AI Fellowship 3기](https://www.sktaifellowship.com) [AI 기반 고 디지털 미디어 복원 기술 개발](https://www.sktaifellowship.com/4222451b-09b3-4627-a013-9732bcdadf57)에 관련된 내용을 업로드합니다.

2021.06.01 ~ 2021.11.15 의 기간동안 연구를 수행하였으며, SKT AI Fellowship 3기 대상을 수상하였습니다.

## 프로젝트 소개
* 연구 내용
  * 오래된 영상물의 화질 개선을 위한 딥러닝 네트워크 모델 분석, 연구, 개발 등
  * Old Image 화질 복원 with AI Tech
* 개발 배경
  * 화질 수준이 매우 열악한 디지털 영상 자료들이 많이 보관되어 있음
  * 이러한 자료들을 복원하는 것은 사회적/경제적 가치 측면에서 매우 의미 있는 일임

## 데이터셋 구성
* 데이터셋 분석
  * 90-00년대 드라마, 영화에는 Blocking, Ringing, Color bleeding 아티팩트들이 존재하는 것으로 확인 
  * 위의 아티팩트들은 압축 과정에서 얻어지는 현상
* 데이터셋 제작
  * 직접 제작한 MPEG2 알고리즘을 기반으로 압축과정을 거침
  * Color bleeding을 더 자세히 구성하기 위하여 압축 과정의 DCT부분을 커스터마이징(SKT 슈퍼노바팀과 특허 출원)
  * Ringing Artifact를 더 넓은 범위로 구현하기 위해 DCT 후 LPF를 사용하여 구성
  * 90년대 영상 내 필름 노이즈와 같은 세로줄 현상을 없애기 위해 세로줄 잡음 생성 후 영상에 추가
  * AWGN 노이즈 레벨 20~30 사이로 랜덤하게 생성하여 잡음 추가

## 제안하는 딥 러닝 모델
* 네트워크 구조
![network](https://user-images.githubusercontent.com/59470033/143054997-4f87fdfc-daf3-4233-bd3c-899d2127c181.png)
  * Multi-progressive Image Restoration(MPRNet)의 Multi-stage, CSFF, SAM을 이용하여 네트워크를 구성
  * CSFF(Cross-Stage Feature Fusion)는 현재 스테이지의 피쳐들을 다음 스테이지로 전달하는 역할, U-net 구조에서 인코더와 디코더 각각의 결과를 다음 스테이지 인코더로 전달
  * SAM(Supervised Attention Module)은 Ground-Truth 이미지와 로스 비교를 통해 Attention Map을 생성하고 유용한 피쳐를 다음 스테이지로 전달
  * 맨 마지막 단에 Up-scale(scale factor=2)를 구성하여 Super-Resolution을 통해 해상도를 높여 더욱 고품질의 영상을 얻고자 함

* Feature Extractor
  *  엣지 모듈과 ASPP 블록으로 구성되어 있음
  *  열화된 이미지에서부터 열화현상이 제거된 엣지를 추출하기 위해 Ground-Truth Edge와 로스 비교를 통해 학습된 엣지를 추출할 수 있도록 엣지 모듈을 설계 -> 열화 현상 제거 시 디테일이 뭉개지거나 다수 잃게 되는 현상을 개선
  *  다양한 수용영역의 정보를 학습 데이터로 이용하기 위해 ASPP 블록을 이용
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/130186086-fc00bfe6-f241-4f3f-ac9a-340548f0889f.png" width="70%" height="70%"></p>


* Channel Attention Based U-net with DC-CAB(Deformable Convolution-Channel Attention Block)  
   * 가중되는 채널에서 각 피쳐들의 집중도를 높이기 위해 CAB 기반의 U-net을 구성
   * 객체의 위치, 중요한 객체를 판단하여 가중치를 주는 Deformable Convolution을 이용한 후 가중되는 채널의 집중도를 위한 CAB 결합
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/143054316-deef9c7e-f9e2-4c5f-99b0-f711c9f838fc.png" width="70%" height="70%"></p>

## 실험 결과
* 목욕탕집 남자들(1995년) 작품에 대한 화질 개선 결과
  * 잡음 및 링잉현상, 반찬에 컬러가 튀는 컬러블리딩 및 무아레 현상을 제거하며 복원
![그림4](https://user-images.githubusercontent.com/59470033/143055789-65da7ae2-1dca-4e8b-b930-03e5ef4b03e1.png)

* 딸부잣집(1994년) 작품에 대한 화질 개선 결과
  * 손 부분에 잡음 및 컬러가 튀는 현상 제거, 머리부분에 링잉현상을 우수하게 제거하며 복원
![그림3](https://user-images.githubusercontent.com/59470033/143055966-774f798a-3a79-4de4-8a5a-95504ac27a9f.png)


