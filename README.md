# SKT_AI_Fellowship

안녕하세요, SKT AI Fellowship 3기 Team SKSAK의 김재엽입니다.

이 레퍼지토리는 [SKT AI Fellowship 3기](https://www.sktaifellowship.com) [AI 기반 고 디지털 미디어 복원 기술 개발](https://www.sktaifellowship.com/4222451b-09b3-4627-a013-9732bcdadf57)에 관련된 내용을 업로드합니다.

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
  * Color bleeding을 더 자세히 구성하기 위하여 압축 과정의 DCT부분을 커스터마이징

## 제안하는 딥 러닝 모델
* 네트워크 구조
![network_overall](https://user-images.githubusercontent.com/59470033/130185260-91185cab-fe4f-4cd7-932e-e5f3a8beb74a.png)
  * Multi-progressive Image Restoration(MPRNet)의 Multi-stage, CSFF, SAM을 이용하여 네트워크를 구성
  * CSFF(Cross-Stage Feature Fusion)는 현재 스테이지의 피쳐들을 다음 스테이지로 전달하는 역할, U-net 구조에서 인코더와 디코더 각각의 결과를 다음 스테이지 인코더로 전달
  * SAM(Supervised Attention Module)은 Ground-Truth 이미지와 로스 비교를 통해 Attention Map을 생성하고 유용한 피쳐를 다음 스테이지로 전달


* Feature Extractor
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/130186086-fc00bfe6-f241-4f3f-ac9a-340548f0889f.png" width="70%" height="70%"></p>
  
  * 엣지 모듈과 ASPP 블록으로 구성되어 있음.
  * 열화된 이미지에서부터 열화현상이 제거된 엣지를 추출하기 위해 Ground-Truth Edge와 로스 비교를 통해 학습된 엣지를 추출할 수 있도록 엣지 모듈을 설계 -> 열화 현상 제거 시 디테일이 뭉개지거나 다수 잃게 되는 현상을 개선
  * 다양한 수용영역의 정보를 학습 데이터로 이용하기 위해 ASPP 블록을 이용


* Channel Attention Based U-net with WRCAB(Wide Receptive Field Channel Attention Block)
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/130187459-4a462023-3d98-453f-9186-c0e7f3ebbfa7.png" width="50%" height="50%"></p>
  
  * 가중되는 채널에서 각 피쳐들의 집중도를 높이기 위해 CAB 기반의 U-net을 구성
  * 다양한 수용 영역의 정보를 학습하기 위해 가장 피쳐의 수가 많은 U-net의 플랫한 부분에 WRCAB를 추가 구성



