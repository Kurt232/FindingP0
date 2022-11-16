import numpy as np  # 调用numpy多维数组包——用来建立BA网格
import matplotlib.pyplot as plt  # 调用matplotlib画图包——生成SEIR模型图
import networkx as nx  # 调用networkx图形包——绘制BA随机分布图
import sys, os
import pandas as pd

# 使用taichi加速
import taichi as ti
ti.init(arch=ti.gpu)

def create_er_graph(argv: list):
    g = nx.random_graphs.erdos_renyi_graph(argv[0], argv[1], argv[2])
    return g

def create_ba_graph(argv: list):
    g = nx.random_graphs.barabasi_albert_graph(argv[0], argv[1], argv[2])
    return g

def create_graph(types_g: str, nodes: int):
    if types_g == 'er':
        g = create_er_graph([nodes, 0.25, None])
    elif types_g == 'ba':
        g = create_ba_graph([nodes, 6, None])
    return nx.to_numpy_array(g), g.size()

# def load_graph_csv(filepath:str):
#     df = pd.read_csv()

@ti.kernel
def map2vec(g: ti.types.ndarray(), e: ti.types.ndarray()):
    n = g.shape[0]
    k = 0
    for i, j in ti.ndrange(n, n): 
        if g[i, j] == 1:
            e[k, 0] = i
            e[k, 1] = j
            k = k + 1

def save_graph_csv(g: np.array, size: int, file_path: str):
    e = np.zeros((2 * size, 2), dtype=int)
    map2vec(g, e)
    df = pd.DataFrame(e)
    df.columns = ['src', 'dst']
    df.to_csv(file_path + '/edges.csv', index=False)

@ti.kernel
def tran_node_state(state: ti.types.ndarray()):
    n, t = state.shape
    for _n, _t in ti.ndrange(n, t):
        if state[_n, _t] == 3:
            state[_n, _t] = 1
        elif state[_n, _t] == 4:
            state[_n, _t] = 2
        else:
            state[_n, _t] = 0

def save_node_csv(state: np.array, file_path:str):
    _, days = state.shape
    df = pd.DataFrame(state)
    df.columns = ['day' + str(x) for x in range(days)]
    df.to_csv(file_path + '/nodes.csv', index=False)

if __name__ == "__main__":
    """
    argv[1] graphname: str
    argv[2] pathname: str
    argv[3] epoch: int 
    argv[4] days: int 
    argv[5] nodes: int 
    argv[6] single_g : [0, 1] 0:False 1: True
    """
    lenargv = len(sys.argv)
    if  lenargv < 5:
        sys.stderr.write('args error')
        sys.exit()
    
    # 入口参数 将会写在/rawdata/下
    type_g = sys.argv[1]
    path = r'./rawdata/'
    path = path + sys.argv[2] + '/'
    # if not os.path.exists(path): 以防错误启动py 覆盖数据
    os.makedirs(path)
    epoch = int(sys.argv[3])
    days = int(sys.argv[4])
    n_node = 1000
    single_g = False
    
    if lenargv >= 6:
        n_node = int(sys.argv[5])
    if lenargv >= 7:
        single_g = '1' == sys.argv[6]
    
    #SEIR 参数 感染率0.2，发病率0.5，康复率0.1
    s2e = ti.field(ti.f32, shape=())
    s2e[None] = 2.5
    e2i = ti.field(ti.f32, shape=())
    e2i[None] = 0.5
    i2r = ti.field(ti.f32, shape=())
    i2r[None] = 0.1

    @ti.kernel
    def spread(g: ti.types.ndarray(), degree: ti.types.ndarray()):
        nodes, days = degree.shape
        for t in ti.ndrange(days-1):
            for i in ti.ndrange(nodes):
                if degree[i, t] == 1:  # 若节点状态为1，即"易感者"
                    lines = 0  # 计算节点附近的邻边数
                    for j in ti.ndrange(nodes):
                        if g[i, j] == 1 and degree[j, t] == 3:
                            lines = lines + 1
                    oops = 1 - (1 - s2e[None]) ** lines
                    p = ti.random()  # 当有n条邻边时，被感染概率为1-（1-w）^n
                    if p < oops:
                        degree[i, t + 1] = 2  # 被感染，更新状态为E
                    #else 仍是 1
                elif degree[i, t] == 2:  # 若节点状态为2，转为I概率为E_to_I
                    p = ti.random()
                    if p < e2i[None]:
                        degree[i, t + 1] = 3 # 若转换为I，更新状态为I
                    else:
                        degree[i, t + 1] = 2
                elif degree[i, t] == 3:  # 若节点状态为3，转为R概率为I_to_R
                    p = ti.random()
                    if p < i2r[None]:
                        degree[i, t + 1] = 4
                    else:
                        degree[i, t + 1] = 3
            ti.lang.runtime_ops.sync()
        

    def epedemic_Simulation(types_g, nodes, t, epochs, file_path, single_g):
        if single_g:
            g, g_size = create_graph(types_g, nodes) 
            save_graph_csv(g, g_size, file_path)

        for i in range(epochs):
            node_path = file_path + str(i) + '/'
            os.makedirs(node_path)
            if not single_g:
                g, g_size = create_graph(types_g, nodes) 
                save_graph_csv(g, g_size, node_path)

            degrees = np.ones((nodes, t),dtype=int) 
            # 生成N个节点数，默认为1，即"易感者"
            iNum = np.random.randint(0, nodes)
            degrees[iNum, 0] = 3  # 随机抽取一位"发病者"
            spread(g, degrees)
            tran_node_state(degrees)
            save_node_csv(degrees, node_path)

    epedemic_Simulation(type_g, n_node, days, epoch, path, single_g)
# 人数为1000，没增加一个点添加6个边，50天实验期, 一个图生成1次
