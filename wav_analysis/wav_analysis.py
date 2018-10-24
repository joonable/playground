import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

all_category = []
folders = os.listdir('./data')
for folder in folders:
    category = []
    wav_files = os.listdir('./data/' + folder)
    for wav in wav_files:
        path = './data/{0}/{1}'.format(folder, wav)
        sample_rate, sample = wavfile.read(path)
        category.append({
            "category": folder,
            "sample_rate": sample_rate,
            "sample": sample
        })
    all_category.append(category)

data = all_category[0][0]
# print(data)
sample_rate, sample = data["sample_rate"], data["sample"]  # ex) 44100, (1238400, 2)
# 44100 => 초당 측정횟수
# 1238400 => 총 측정 횟수
# 2 => 채널수 (이어폰 생각하면..)
# 1238400 / 44100 = 28,  실제 wav파일 28초

sample_mono = np.mean(sample, 1)  # 양쪽 채널를 단일 채널로 변경 → (1238400, 1)
# 1 초당 44100 번 측정했으므로
# 1 초는 30 frame 으로 변경하려면
# 44100 / 30 = 1470
# 1238400 / 1470 = 843

line_space_sample = np.linspace(0, 1238400, num=1238400, endpoint=False)  # 측정횟수
line_space_frame = np.linspace(0, 843, num=843, endpoint=False)  # frame

sample_mono_frame = [value for index, value in enumerate(sample_mono) if index % 1470 == 0]
# print(len(sample_mono_frame))  # 843


# 측정 단위로 본 진폭 VS 시간
f, ax = plt.subplots()
ax.plot(line_space_sample, sample_mono)
ax.set_xlabel('sample_num')
ax.set_ylabel('amplitude(power)')  # 진폭
plt.show()

# frame단위로 본 진폭 VS 시간
f, ax = plt.subplots()
ax.plot(line_space_frame, sample_mono_frame)
ax.set_xlabel('frame')
ax.set_ylabel('amplitude(power)')  # 진폭
plt.show()


# 1 초 = 30 frame
# 고맙습니다 ^^ 44100HZ
# 신호를 30HZ 신호로 낮추면서 max(amplitude) 도 많이 줄었네요.
# 아무래도 포인트간 거리가 넓어지다 보니까 포인트 사이에 있던 maximum 값들이 날아가는 것 같습니다.
# interpolation을 해도 아마 smoothing되는 것처럼 maximum값들이 사라질 것 같고.. 므튼 잘 봤습니당~ ^^