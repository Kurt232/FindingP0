import numpy as np  
import matplotlib.pyplot as plt 
import networkx as nx  
import sys, os
import pandas as pd
import argparse

# 使用taichi加速
import taichi as ti
ti.init(arch=ti.cpu)

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
    # 若不序列化 会在边后面生成大量的0
    ti.loop_config(serialize=True) # 似乎需要序列化
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
    argv[7] start_day: int default : 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=str, choices=['er', 'ba'], help='graph name')
    parser.add_argument("path", type=str,help='input subpath')
    parser.add_argument("epoch", type=int)
    parser.add_argument("days", type=int)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--single", type=int, choices=[False, True], default=False)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    
    # 入口参数 将会写在/rawdata/下
    path = r'./rawdata/'
    path = path + args.path + '/'
    if args.single:
        args.start = 0
        print("单图生成 不能分段生成")
    
    # if not os.path.exists(path): 以防错误启动py 覆盖数据
    if os.path.exists(path):
        if args.start > 0:
            print(f"从{args.start}开始生成")
        else:
            print("存在路径冲突")
        if os.path.exists(path+ str(args.start) + '/'):
            print("存在覆盖冲突")
    else:
        os.makedirs(path)
    
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
        ti.loop_config(serialize=True) #TODO
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
                elif degree[i, t] == 4:
                    degree[i, t + 1] = 4
            ti.lang.runtime_ops.sync()
        

    def epedemic_Simulation(types_g, nodes, t, epochs, file_path, single_g, start_d):
        if single_g:
            g, g_size = create_graph(types_g, nodes) 
            save_graph_csv(g, g_size, file_path)

        for i in range(epochs):
            if i%100 == 0:
                print(f"epoch: {i}")
            node_path = file_path + str(i + start_d) + '/'
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

    epedemic_Simulation(args.graph, args.nodes, args.days, args.epoch, path, args.single, args.start)
# 人数为1000，没增加一个点添加6个边，50天实验期, 一个图生成1次
