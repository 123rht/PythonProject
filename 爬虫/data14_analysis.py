import chardet as chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def tem_curve(data):
    """温度曲线绘制"""
    date = list(data['日期'])
    tem_low = list(data['最低气温'])
    tem_high = list(data['最高气温'])
    for i in range(0, 14):
        if math.isnan(tem_low[i]) == True:
            tem_low[i] = tem_low[i - 1]
        if math.isnan(tem_high[i]) == True:
            tem_high[i] = tem_high[i - 1]

    tem_high_ave = sum(tem_high) / 14  # 求平均高温
    tem_low_ave = sum(tem_low) / 14  # 求平均低温

    tem_max = max(tem_high)
    tem_max_date = tem_high.index(tem_max)  # 求最高温度
    tem_min = min(tem_low)
    tem_min_date = tem_low.index(tem_min)  # 求最低温度

    x = range(1, 15)
    plt.figure(1)
    plt.plot(x, tem_high, color='red', label='高温')  # 画出高温度曲线
    plt.scatter(x, tem_high, color='red')  # 点出每个时刻的温度点
    plt.plot(x, tem_low, color='blue', label='低温')  # 画出低温度曲线
    plt.scatter(x, tem_low, color='blue')  # 点出每个时刻的温度点

    plt.plot([1, 15], [tem_high_ave, tem_high_ave], c='black', linestyle='--')  # 画出平均温度虚线
    plt.plot([1, 15], [tem_low_ave, tem_low_ave], c='black', linestyle='--')  # 画出平均温度虚线
    plt.legend()
    plt.text(tem_max_date + 0.15, tem_max + 0.15, str(tem_max), ha='center', va='bottom', fontsize=10.5)  # 标出最高温度
    plt.text(tem_min_date + 0.15, tem_min + 0.15, str(tem_min), ha='center', va='bottom', fontsize=10.5)  # 标出最低温度
    plt.xticks(x)
    plt.title('未来14天高温低温变化曲线图')
    plt.xlabel('未来天数/天')
    plt.ylabel('摄氏度/℃')
    plt.show()


def change_wind(wind):
    """改变风向"""
    for i in range(0, 14):
        if wind[i] == "北风":
            wind[i] = 90
        elif wind[i] == "南风":
            wind[i] = 270
        elif wind[i] == "西风":
            wind[i] = 180
        elif wind[i] == "东风":
            wind[i] = 360
        elif wind[i] == "东北风":
            wind[i] = 45
        elif wind[i] == "西北风":
            wind[i] = 135
        elif wind[i] == "西南风":
            wind[i] = 225
        elif wind[i] == "东南风":
            wind[i] = 315
    return wind


def wind_radar(data):
    """风向雷达图"""
    wind1 = list(data['风向1'])
    wind2 = list(data['风向2'])
    wind_speed = list(data['风级'])
    wind1 = change_wind(wind1)
    wind2 = change_wind(wind2)

    degs = np.arange(45, 361, 45)
    temp = []
    for deg in degs:
        speed = []
        # 获取 wind_deg 在指定范围的风速平均值数据
        for i in range(0, 14):
            if wind1[i] == deg:
                speed.append(wind_speed[i])
            if wind2[i] == deg:
                speed.append(wind_speed[i])
        if len(speed) == 0:
            temp.append(0)
        else:
            temp.append(sum(speed) / len(speed))
    print(temp)
    N = 8
    theta = np.arange(0. + np.pi / 8, 2 * np.pi + np.pi / 8, 2 * np.pi / 8)
    # 数据极径
    radii = np.array(temp)
    # 绘制极区图坐标系
    plt.axes(polar=True)
    # 定义每个扇区的RGB值（R,G,B），x越大，对应的颜色越接近蓝色
    colors = [(1 - x / max(temp), 1 - x / max(temp), 0.6) for x in radii]
    plt.bar(theta, radii, width=(2 * np.pi / N), bottom=0.0, color=colors)
    plt.title('未来14天风级图', x=0.2, fontsize=20)
    plt.show()


def weather_pie(data):
    """绘制天气饼图"""
    weather = list(data['天气'])
    dic_wea = {}
    for i in range(0, 14):
        if weather[i] in dic_wea.keys():
            dic_wea[weather[i]] += 1
        else:
            dic_wea[weather[i]] = 1
    print(dic_wea)
    explode = [0.01] * len(dic_wea.keys())
    color = ['lightskyblue', 'silver', 'yellow', 'salmon', 'grey', 'lime', 'gold', 'red', 'green', 'pink']
    plt.pie(dic_wea.values(), explode=explode, labels=dic_wea.keys(), autopct='%1.1f%%', colors=color)
    plt.title('未来14天气候分布饼图')
    plt.show()

def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']

def main():
    get_encoding('weather14.csv')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    data14 = pd.read_csv('weather14.csv', encoding='utf_8_sig')



    tem_curve(data14)
    wind_radar(data14)
    weather_pie(data14)



if __name__ == '__main__':
    main()
