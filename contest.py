import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import joblib
from datetime import datetime

# 設定參數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# 合併所有 17 個檔案的數據
def load_all_data():
    data_path = os.getcwd() + r'\LSTM+迴歸分析(比賽用)\LSTM+迴歸分析(比賽用)\ExampleTrainData(AVG)'
    all_files = glob.glob(data_path + r'\AvgDATA_*.csv')
    all_data = pd.concat([pd.read_csv(file, encoding='utf-8') for file in all_files], ignore_index=True)
    return all_data

# 載入數據
all_data = load_all_data()

# 分別準備 LSTM 和回歸模型需要的數據
LSTM_X_data = all_data[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
Regression_X_train = all_data[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values
Regression_y_train = all_data[['Power(mW)']].values

# 正規化數據
LSTM_MinMaxModel = MinMaxScaler().fit(LSTM_X_data)
LSTM_X_data_scaled = LSTM_MinMaxModel.transform(LSTM_X_data)

# 準備 LSTM 的訓練數據
X_train, y_train = [], []

for i in range(LookBackNum, len(LSTM_X_data_scaled)):
    X_train.append(LSTM_X_data_scaled[i - LookBackNum:i, :])
    y_train.append(LSTM_X_data_scaled[i, :])

X_train = np.array(X_train)
y_train = np.array(y_train)

# 重新設定形狀
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

# 建置 LSTM 模型
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64))
    model.add(Dense(units=5))
    model.compile(optimizer='adam', loss='mean_absolute_error')  # 改用 MAE
    return model

# 訓練 LSTM 模型
lstm_model = build_lstm_model()
lstm_model.fit(X_train, y_train, epochs=100, batch_size=128)

# 保存 LSTM 模型
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
lstm_model.save(f'WeatherLSTM_{NowDateTime}.h5')
print('LSTM Model Saved')

# 訓練 XGBoost 模型
XGBModel = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
XGBModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

# 保存 XGBoost 模型
joblib.dump(XGBModel, f'WeatherXGBoost_{NowDateTime}.joblib')
print('XGBoost Model Saved')

# 打印 XGBoost 模型性能
print("XGBoost R^2:", XGBModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))

#============================預測數據============================

# 載入模型
lstm_model = load_model(f'WeatherLSTM_{NowDateTime}.h5')
XGBModel = joblib.load(f'WeatherXGBoost_{NowDateTime}.joblib')

# 讀取測試資料的 CSV 檔案，並提取要預測的序號
DataName = os.getcwd() + r'\LSTM+迴歸分析(比賽用)\LSTM+迴歸分析(比賽用)\ExampleTestData\upload.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

# 初始化變量以儲存參考資料和預測結果
inputs = []
PredictOutput = []
PredictPower = []

count = 0
while count < len(EXquestion):
    print('count : ', count)
    LocationCode = int(EXquestion[count])  # 提取裝置代號
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = '0' + strLocationCode

    # 讀取該裝置的歷史數據，並提取12個時間步的數據
    DataName = os.getcwd() + f'\LSTM+迴歸分析(比賽用)\ExampleTrainData(IncompleteAVG)\IncompleteAvgDATA_{strLocationCode}.csv'
    SourceData = pd.read_csv(DataName, encoding='utf-8')
    ReferTitle = SourceData[['Serial']].values
    ReferData = SourceData[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']].values

    inputs = []
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
            TempData = ReferData[DaysCount].reshape(1, -1)
            TempData = LSTM_MinMaxModel.transform(TempData)
            inputs.append(TempData)

    for i in range(ForecastNum):
        if i > 0:
            inputs.append(PredictOutput[i-1].reshape(1, 5))

        X_test = []
        X_test.append(inputs[0+i:LookBackNum+i])
        NewTest = np.array(X_test)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 5))

        predicted = lstm_model.predict(NewTest)
        PredictOutput.append(predicted)
        PredictPower.append(np.round(XGBModel.predict(predicted), 2).flatten())

    count += 48

# 輸出結果到 CSV
df = pd.DataFrame({
    '序號': EXquestion.flatten(),
    '答案': [round(float(x), 2) for x in PredictPower]
})
output_file = 'upload.csv'
df.to_csv(output_file, index=False)
print('Output CSV File Saved')
