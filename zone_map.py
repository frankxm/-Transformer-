
#folium-聚合散点地图
import folium
from folium import plugins
import os
# 在py中临时添加的环境变量,防止用kmeans警告
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from folium import FeatureGroup, Marker, LayerControl
def get_map():
    # 地震区域分块
    # # 设置超参数
    N_CLUSTERS = 9  # 类簇的数量
    MARKERS = ['*', 'v', '+', '^', 's', 'x', 'o', '<', '>']  # 标记样式（绘图）
    COLORS = ['pink', 'g', 'm', 'c', 'y', 'b', 'orange', 'k', 'yellow']  # 标记颜色（绘图）

    # 海原 银川 临夏 高台
    centers = np.array([[105.61, 36.51], [105.93, 38.61], [103.2, 35.6], [99.86, 39.4]])

    df = pd.read_csv('./Data/Area1_3~10.csv', encoding='gb2312')
    x = df[['纬度(°)', '经度(°)']]
    x_np = np.array(x)  # 将x转化为numpy数组

    # 模型构建
    model = KMeans(N_CLUSTERS, random_state=2)  # 构建聚类器
    model.fit(x)  # 训练聚类器

    labels = model.labels_  # 获取聚类标签
    print('聚类标签', labels, len(labels), type(labels))

    unique, count = np.unique(labels, return_counts=True)
    data_count = dict(zip(unique, count))
    print('各类别出现个数:', data_count)

    # 获取地图基底
    def getMapObject(baseSource=2, centerLoc=[0, 0], baseLayerTitle='baseLayer'):  # 0:googleMap, 1: 高德地图，2:腾讯地图
        if baseSource == 0:
            m = folium.Map(location=centerLoc,
                           min_zoom=0,
                           max_zoom=19,
                           zoom_start=5,
                           control=False,
                           control_scale=True
                           )

        elif baseSource == 1:
            # 下面的程式将使用高德地图作为绘图的基底
            m = folium.Map(location=centerLoc,
                           zoom_start=5,
                           control_scale=True,
                           control=False,
                           tiles=None
                           )

            folium.TileLayer(
                tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                attr="&copy; <a href=http://ditu.amap.com/>高德地图</a>",
                min_zoom=0,
                max_zoom=19,
                control=True,
                show=True,
                overlay=False,
                name=baseLayerTitle
                ).add_to(m)
        else:
            # 下面的程式将使用腾讯地图作为绘图的基底
            m = folium.Map(location=centerLoc,
                           zoom_start=5,
                           control_scale=True,
                           control=False,
                           tiles=None
                           )

            folium.TileLayer(tiles='https://rt0.map.gtimg.com/tile?z={z}&x={x}&y={-y}',
                             attr="&copy; <a href=http://map.qq.com/>腾讯地图</a>",
                             min_zoom=0,
                             max_zoom=19,
                             control=True,
                             show=True,
                             overlay=False,
                             name=baseLayerTitle
                             ).add_to(m)
        return m

    plotmap1 = getMapObject(centerLoc=[36.47, 101.79])
    # 海原 银川 临夏 高台
    centers = [[36.51, 105.61], [38.61, 105.93], [35.6, 103.2], [39.4, 99.86]]
    folium.Marker(location=[36.51, 105.61], popup=folium.Popup("海原台 纬度36.51 经度105.61", max_width=100),
                  icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
    folium.Marker(location=[38.61, 105.93], popup=folium.Popup("银川台 纬度38.61 经度105.93", max_width=100),
                  icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
    folium.Marker(location=[35.6, 103.2], popup=folium.Popup("临夏台 纬度35.6 经度103.2", max_width=100),
                  icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
    folium.Marker(location=[39.4, 99.86], popup=folium.Popup("高台 纬度39.4 经度99.86", max_width=100),
                  icon=folium.Icon(icon='cloud', color='red'), tooltip='观测点').add_to(plotmap1)
    # 添加经纬度
    plotmap1.add_child(folium.LatLngPopup())

    colors = ['pink', 'green', 'magenta', 'cyan', 'brown', 'blue', 'orange', 'black', 'yellow']
    area_list=[]
    for i in range(0, 9):
        color = colors[i]
        # 定位不同区域的坐标
        loc_ind = np.where(labels == i)
        # 获取np.where返回值中符合条件的个数
        # print('符合标签的个数为 ',len(loc_ind[0]))
        arr = x_np[loc_ind]
        # 计算外接凸图案的边界点
        hull = ConvexHull(arr)
        # 获取原数据中外围边界点的索引
        hull1 = hull.vertices.tolist()
        hull1.append(hull1[0])
        arr = arr[hull1]
        location = [[ar[0], ar[1]] for ar in arr]

        nd_location=np.array(location)
        min_xy=np.min(nd_location,axis=0)
        max_xy=np.max(nd_location,axis=0)
        area_xy=np.vstack((min_xy,max_xy))
        area_list.append(area_xy)
        plotmap1.add_child(plugins.MarkerCluster(location))
        folium.Polygon(
            locations=location,
            popup=folium.Popup(f'区域{i}', max_width=200),
            color=color,  # 线颜色
            fill=True,  # 是否填充
            weight=3,  # 边界线宽
        ).add_to(plotmap1)

    LayerControl().add_to(plotmap1)
    plotmap1.save('map.html')

if __name__ == '__main__':
    get_map()