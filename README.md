# BK21 Hackathon

### python code
-	data_preprocess.py : EDA 및 데이터 전처리
-	Correlation.py : 변수 간 Correlation 확인
-	et_utils.py : timeonly_et.py, time_weather_et.py 돌리는 데 필요한 파일
-	timeonly_et.py : 시간 데이터만 활용한 Extra Trees 모델 학습 및 평가
-	time_weather_et.py : 시간, 날씨 데이터 활용한 Extra Trees 모델 학습 및 평가
-	DNN_utils.py : DNN.py 돌리는 데 필요한 파일
-	DNN.py : DNN 모델 학습 및 평가
-	metric.py : 모델 간 비교

### Image(Folder)
-	timeonly_et(Folder) : 시간 데이터만 활용한 Extra Trees 모델의 각 Inverter별 성능
-	time_weather_et(Folder) : 시간, 날씨 데이터 활용한 Extra Trees 모델의 각 Inverter별 성능
-	DNN(Folder)
  - compare_real_pred(Folder) : DNN 모델의 각 Inverter별 성능
  - Loss(Folder) : DNN 학습과정 동안의 train, val Loss
