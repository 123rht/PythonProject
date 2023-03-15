import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def tem_curve(data):
    """温度曲线绘制"""
    hour = list(data['小时'])
    tem = list(data['温度'])
    for i in range(0, 24):
        if math.isnan(tem[i]) == True:
            tem[i] = tem[i - 1]
    tem_ave = sum(tem) / 24  # 求平均温度
    tem_max = max(tem)
    tem_max_hour = hour[tem.index(tem_max)]  # 求最高温度
    tem_min = min(tem)
    tem_min_hour = hour[tem.index(tem_min)]  # 求最低温度
    x = []
    y = []
    for i in range(0, 24):
        x.append(i)
        y.append(tem[hour.index(i)])
    plt.figure(1)
    plt.plot(x, y, color='red', label='温度')  # 画出温度曲线
    plt.scatter(x, y, color='red')  # 点出每个时刻的温度点
    plt.plot([0, 24], [tem_ave, tem_ave], c='blue', linestyle='--', label='平均温度')  # 画出平均温度虚线
    plt.text(tem_max_hour + 0.15, tem_max + 0.15, str(tem_max), ha='center', va='bottom', fontsize=10.5)  # 标出最高温度
    plt.text(tem_min_hour + 0.15, tem_min + 0.15, str(tem_min), ha='center', va='bottom', fontsize=10.5)  # 标出最低温度
    plt.xticks(x)
    plt.legend()
    plt.title('一天温度变化曲线图')
    plt.xlabel('时间/h')
    plt.ylabel('摄氏度/℃')
    plt.savefig(r'E:10.png', dpi=1000, bbox_inches='tight')  # 保存至本地
    plt.show()



def hum_curve(data):
    """相对湿度曲线绘制"""
    hour = list(data['小时'])
    hum = list(data['相对湿度'])
    for i in range(0, 24):
        if math.isnan(hum[i]) == True:
            hum[i] = hum[i - 1]
    hum_ave = sum(hum) / 24  # 求平均相对湿度
    hum_max = max(hum)
    hum_max_hour = hour[hum.index(hum_max)]  # 求最高相对湿度
    hum_min = min(hum)
    hum_min_hour = hour[hum.index(hum_min)]  # 求最低相对湿度
    x = []
    y = []
    for i in range(0, 24):
        x.append(i)
        y.append(hum[hour.index(i)])
    plt.figure(2)
    plt.plot(x, y, color='blue', label='相对湿度')  # 画出相对湿度曲线
    plt.scatter(x, y, color='blue')  # 点出每个时刻的相对湿度
    plt.plot([0, 24], [hum_ave, hum_ave], c='red', linestyle='--', label='平均相对湿度')  # 画出平均相对湿度虚线
    plt.text(hum_max_hour + 0.15, hum_max + 0.15, str(hum_max), ha='center', va='bottom', fontsize=10.5)  # 标出最高相对湿度
    plt.text(hum_min_hour + 0.15, hum_min + 0.15, str(hum_min), ha='center', va='bottom', fontsize=10.5)  # 标出最低相对湿度
    plt.xticks(x)
    plt.legend()
    plt.title('一天相对湿度变化曲线图')
    plt.xlabel('时间/h')
    plt.ylabel('百分比/%')
    plt.show()


def air_curve(data):
    """空气质量曲线绘制"""
    hour = list(data['小时'])
    air = list(data['空气质量'])
    print(type(air[0]))
    for i in range(0, 24):
        if math.isnan(air[i]) == True:
            air[i] = air[i - 1]
    air_ave = sum(air) / 24  # 求平均空气质量
    air_max = max(air)
    air_max_hour = hour[air.index(air_max)]  # 求最高空气质量
    air_min = min(air)
    air_min_hour = hour[air.index(air_min)]  # 求最低空气质量
    x = []
    y = []
    for i in range(0, 24):
        x.append(i)
        y.append(air[hour.index(i)])
    plt.figure(3)

    for i in range(0, 24):
        if y[i] <= 50:
            plt.bar(x[i], y[i], color='lightgreen', width=0.7)  # 1等级
        elif y[i] <= 100:
            plt.bar(x[i], y[i], color='wheat', width=0.7)  # 2等级
        elif y[i] <= 150:
            plt.bar(x[i], y[i], color='orange', width=0.7)  # 3等级
        elif y[i] <= 200:
            plt.bar(x[i], y[i], color='orangered', width=0.7)  # 4等级
        elif y[i] <= 300:
            plt.bar(x[i], y[i], color='darkviolet', width=0.7)  # 5等级
        elif y[i] > 300:
            plt.bar(x[i], y[i], color='maroon', width=0.7)  # 6等级
    plt.plot([0, 24], [air_ave, air_ave], c='black', linestyle='--')  # 画出平均空气质量虚线
    plt.text(air_max_hour + 0.15, air_max + 0.15, str(air_max), ha='center', va='bottom', fontsize=10.5)  # 标出最高空气质量
    plt.text(air_min_hour + 0.15, air_min + 0.15, str(air_min), ha='center', va='bottom', fontsize=10.5)  # 标出最低空气质量
    plt.xticks(x)
    plt.title('一天空气质量变化曲线图')
    plt.xlabel('时间/h')
    plt.ylabel('空气质量指数AQI')
    plt.show()


def wind_radar(data):
    """风向雷达图"""
    wind = list(data['风力方向'])
    wind_speed = list(data['风级'])
    for i in range(0, 24):
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
    degs = np.arange(45, 361, 45)
    temp = []
    for deg in degs:
        speed = []
        # 获取 wind_deg 在指定范围的风速平均值数据
        for i in range(0, 24):
            if wind[i] == deg:
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
    plt.title('一天风级图', x=0.2, fontsize=20)
    plt.show()


def calc_corr(a, b):
    """计算相关系数"""
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    corr_factor = cov_ab / sq
    return corr_factor


def corr_tem_hum(data):
    """温湿度相关性分析"""
    tem = data['温度']
    hum = data['相对湿度']
    plt.scatter(tem, hum, color='blue')
    plt.title("温湿度相关性分析图")
    plt.xlabel("温度/℃")
    plt.ylabel("相对湿度/%")
    plt.text(20, 40, "相关系数为：" + str(calc_corr(tem, hum)), fontdict={'size': '10', 'color': 'red'})
    plt.show()
    print("相关系数为：" + str(calc_corr(tem, hum)))


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    data1 = pd.read_csv('weather1.csv', encoding='utf_8_sig')
    print(data1)
    tem_curve(data1)
    hum_curve(data1)
    air_curve(data1)
    wind_radar(data1)
    corr_tem_hum(data1)


if __name__ == '__main__':
    main()
