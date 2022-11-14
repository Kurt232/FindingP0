import numpy as np  # 调用numpy多维数组包——用来建立ER网格
import matplotlib.pyplot as plt  # 调用matplotlib画图包——生成SEIR模型图
import random  # 调用random随机包——随机感染
import networkx as nx  # 调用networkx图形包——绘制ER随机分布图
import sys, os

def create_ER(N, p, file_path: str, random_seed=None):  # 利用numpy生成N*N[0，1]的随机二维数组，再用np（where）条件赋值
    ER = nx.random_graphs.erdos_renyi_graph(N, p, seed = random_seed)
    pos = nx.spring_layout(ER)
    nx.draw(ER, pos, with_labels = False, node_size = 30)
    plt.savefig(file_path + 'ER网格图.png')
    return nx.to_numpy_array(ER)


def save_ER_to_file(file_path, ERmap):  # 将ER邻边连接随机网格写入txt文档
    file = open(file_path + 'edges.csv', 'w+')
    file.write('src,dst\n')
    for i in range(len(ERmap)):  # 打开文档，若无将自动创建
        a = ERmap[i]
        for j in range(len(ERmap)):
            if a[j] == 1:  # 先取第一行，再取各列
                file.write(str(i) + ', ' + str(j) + '\n')# TODO 边权默认为1
                # file.write(str(j) + ', ' + str(i) + '\n')# TODO 边权默认为1
    file.close()

def save_ER_new_to_file(degrees, days: int, node_path):  # 节点状态
    file = open(node_path + 'nodes' + str(days) + '.csv', 'w+')
    file.write('state,days\n')
    for i in range(degrees.size):  # 打开文档，若无将自动创建
        if degrees[i] == 3:
            t = '1'  # 先取第一行，再取各列
        elif degrees[i] == 4:
            t = '2'
        else:
            t = '0'
        file.write(str(t) + ', ' + str(days) + '\n')
    file.close()


def calculateDegreeDistribution(file_path, ERmap):
    avedegree = 0.0  # 计算度分布
    identify = 0.0  # 若算法正确，度概率分布总和应为0
    p_degree = np.zeros((len(ERmap)), dtype=float)
    # statistic下标为度值
    # (1)先计数该度值的量
    # (2)再除以总节点数N得到比例
    degree = np.zeros(len(ERmap), dtype=int)
    # degree用于存放各个节点的度之和
    for i in range(len(ERmap)):
        for j in range(len(ERmap)):
            degree[i] = degree[i] + ERmap[i][j]
            # 汇总各个节点的度之和
    for i in range(len(ERmap)):
        avedegree += degree[i]
        # 汇总每个节点的度之和
    print('该模型平均度为\t' + str(avedegree / len(ERmap)))
    # 计算平均度
    for i in range(len(ERmap)):
        p_degree[degree[i]] = p_degree[degree[i]] + 1
        # 先计数该度值的量
    for i in range(len(ERmap)):  # 再除以总节点数N得到比例
        p_degree[i] = p_degree[i] / len(ERmap)
        identify = identify + p_degree[i]
        # 将所有比例相加，应为1
    identify = int(identify)
    plt.figure(figsize=(10, 4), dpi=120)
    # 绘制度分布图
    plt.xlabel('$Degree$', fontsize=21)
    # 横坐标标注——Degrees
    plt.ylabel('$P$', fontsize=26)
    # 纵坐标标注——P
    plt.plot(list(range(len(ERmap))), list(p_degree), '-*', markersize=15, label='度', color="#ff9c00")
    # 自变量为list(range(N)),因变量为list(p_degree)
    # 图形标注选用星星*与线条-，大小为15，标注图例为度，颜色是水果橘

    plt.xlim([0, 12])  # 给x轴设限制值
    plt.ylim([-0.05, 0.5])  # 给y轴设限制值
    plt.xticks(fontsize=20)  # 设置x轴的字体大小为21
    plt.yticks(fontsize=20)  # 设置y轴的字体大小为21
    plt.legend(fontsize=21, numpoints=1, fancybox=True, ncol=1)
    plt.savefig(file_path + '度分布图.pdf')
    plt.show()  # 展示图片
    print('算法正常运行则概率之和应为1 当前概率之和=\t' + str(identify))
    # 用于测试算法是否正确
    f = open(file_path + '度分布.txt', 'w+')
    # 将度分布写入文件名为度分布.txt中
    # 若磁盘中无此文件将自动新建
    for i in range(len(ERmap)):
        f.write(str(i))  # 先打印度值、再打印度的比例
        f.write(' ')
        s = str(p_degree[i])  # p_degree[i]为float格式，进行转化才能书写
        f.write(s)  # 再打印度的比例
        f.write('\n')  # 换行
    f.close()
    # 这里正式模拟SEIR传播，本程序将通过每个节点的不同值来表示不同状态


# 1. S(Susceptible) 为 "易感者"，        # 2. E(Explosed)    为 "潜伏者"，无感染力
# 3. I(Infected)    为 "发病者"，有感染力  # 4. R(Recovered)   为 "康复者"，无感染力,不会再被感染
def spread(ERmap, S_to_E, E_to_I, to_R, degree):
    post_degree = np.array(degree)
    for i in range(degree.size):
        if degree[i] == 1:  # 若节点状态为1，即"易感者"
            lines = 0  # 计算节点附近的邻边数
            for j in range(degree.size):
                if ERmap[i, j] == 1 and degree[j] == 3:
                    lines = lines + 1
            oops = 1 - (1 - S_to_E) ** lines
            p = random.random()  # 当有n条邻边时，被感染概率为1-（1-w）^n
            if p < oops:
                post_degree[i] = 2  # 被感染，更新状态为E
        elif degree[i] == 2:  # 若节点状态为2，转为I概率为E_to_I
            p = random.random()
            if p < E_to_I:
                post_degree[i] = 3  # 若转换为I，更新状态为I
        elif degree[i] == 3:  # 若节点状态为3，转为R概率为I_to_R
            p = random.random()
            if p < to_R:
                post_degree[i] = 4
    return post_degree  # 导出传播后节点状态


def epedemic_Simulation(N, p, S_to_E, E_to_I, to_R, t, epochs, file_path):
    ERmap = create_ER(N, p, file_path)  # 生成ER网格
    save_ER_to_file(file_path, ERmap)  # 存储ER网格
    calculateDegreeDistribution(file_path, ERmap)
    # 计算度分布、存储度分布概率结果，并显示度分布图
    # 重复实验次数 Rp 为 100
    Rp = 1  # 概率传播有一定误差，这里重复实验100次
    # 建立四个数组，准备存放不同t时间的节点数
    S = np.array([1 for i in range(t)])
    E = np.array([1 for i in range(t)])
    I = np.array([1 for i in range(t)])
    R = np.array([1 for i in range(t)])

    for i  in range(epochs):
        node_path = file_path + str(i) + '/'
        # if not os.path.exists(node_path): #以放错误启动 覆盖
        os.makedirs(node_path)
        for a in range(Rp):  # 重复实验次数Rp次，利用for套内循环
            degrees = np.array([1 for i in range(N)])
            # 生成N个节点数，默认为1，即"易感者"
            iNum = random.randint(0, N - 1)
            degrees[iNum] = 3  # 随机抽取一位"发病者"
            Ss = []
            Ee = []
            Ii = []
            Rr = []
            save_ER_new_to_file(degrees, 0, node_path)
            for i in range(t):
                if i != 0:
                    degrees = spread(ERmap, S_to_E, E_to_I, to_R, degrees)
                    Ss.append(np.where(np.array(degrees) == 1, 1, 0).sum())
                    Ee.append(np.where(np.array(degrees) == 2, 1, 0).sum())
                    Ii.append(np.where(np.array(degrees) == 3, 1, 0).sum())
                    Rr.append(np.where(np.array(degrees) == 4, 1, 0).sum())
                    save_ER_new_to_file(degrees, i, node_path)
                else:
                    Ss.append(np.where(np.array(degrees) == 1, 1, 0).sum())
                    Ee.append(np.where(np.array(degrees) == 2, 1, 0).sum())
                    Ii.append(np.where(np.array(degrees) == 3, 1, 0).sum())
                    Rr.append(np.where(np.array(degrees) == 4, 1, 0).sum())
            S = S + Ss  # 将每次重复实验数据求和
            E = E + Ee
            I = I + Ii
            R = R + Rr
        print(S)
        for i in range(t):
            S[i] /= Rp  # 求Rp次实验后的平均值
            E[i] /= Rp
            I[i] /= Rp
            R[i] /= Rp
        print(S)  # 检验实验是否正常
        # plt作图
        plt.plot(S, color='darkblue', label='Susceptible', marker='.')
        plt.plot(E, color='orange', label='Exposed', marker='.')
        plt.plot(I, color='red', label='Infection', marker='.')
        plt.plot(R, color='green', label='Recovery', marker='.')
        plt.title('SEIR Model')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Number')
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 100  # 分辨率
        plt.savefig(node_path + 'SEIR曲线图.png')
        # 存储图片
        plt.show()

if __name__ == "__main__":

    if len(sys.argv) > 2:
        sys.stderr.write('only one arg of filename')
        sys.exit()
    # 入口参数 将会写在/rawdata/下
    path = r'./rawdata/'

    path = path + sys.argv[1] + '/'
    # if not os.path.exists(path): 以防错误启动py 覆盖数据
    os.makedirs(path)
    epedemic_Simulation(1000, 0.012, 0.25, 0.6, 0.1, 50, 1, path)
# 人数为1000，邻边结边率为0.006，感染率0.25，发病率0.5，康复率0.1，100天实验期
