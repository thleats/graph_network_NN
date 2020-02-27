import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import itertools
import matplotlib.mlab as ml


plt.close('all')

#function to remove the longest edge from the shortest path
def remove_edge_path(path_edges,G):
    edges=[]
    edges_orig=[]
    for things in path_edges:
        info_orig=(G.get_edge_data(things[0],things[1]))
        edges_orig.append(info_orig['weight'])
        if things[0]>3 and things[1]>3:
            info = G.get_edge_data(things[0],things[1])
            edges.append(info['weight'])
        else:
            pass
    edges_np = np.asarray(edges)#convert to np array
    idx = np.argmax(edges_np)#this is the index in the path minus the pins
    idx_edge = edges_orig.index(edges_np[idx])#this is the index in the path
        
    G.remove_edge(path_edges[idx_edge][0],path_edges[idx_edge][1])
    return(G)

#function to build the graph
def build_graph(G,coords,radii_pin,radii_fiber):
    positions={}
    for i in tqdm(range(len(coords))):
        positions[i] = coords[i,:].tolist()
        G.add_node(i,pos=tuple(positions[i]))
        d=np.sqrt(np.square(coords[i,0]-coords[:,0])+np.square(coords[i,1]-coords[:,1]))
        if i<4:
            log = d<radii_pin
        else:
            log = d<radii_fiber   
        idxs=np.where(log)[0]
        weights = d[idxs]
        for j in range(len(idxs)):
            if i!=idxs[j]:
                G.add_edges_from([(i,idxs[j],{'weight':weights[j]})])
    return(G)

#plot the graph
def plot_graph(G,path,path_edges,plot_path=0):

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos,alpha=0.4)
    
    if plot_path==1:
        nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')
        nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.show()
    
def connections(pin_idxs,cs):
    poss_conn_all = np.asarray(list(itertools.combinations(pin_idxs,2)))
    poss_conn_undesired=[]
    for i in range(len(poss_conn_all)):
        for j in range(len(cs)-1):
            if poss_conn_all[i,0]==cs[j,0] and poss_conn_all[i,1]==cs[j,1] or poss_conn_all[i,1]==cs[j,0] and poss_conn_all[i,0]==cs[j,1] or poss_conn_all[i,0]==cs[j+1,0] and poss_conn_all[i,1]==cs[j+1,1] or poss_conn_all[i,1]==cs[j+1,0] and poss_conn_all[i,0]==cs[j+1,1]:
                pass
            else:
                poss_conn_undesired.append((poss_conn_all[i,0],poss_conn_all[i,1]))
    return(poss_conn_undesired)

def check_cs(cs,G):
    for j in range(len(cs)):
        if nx.has_path(G,cs[j,0],cs[j,1]):
            stop=0
        else:
            print('broke desired path')
            stop=1
            return(stop)
    return(stop)
    
def pare_network(conns,G_temp,alpha,cs):
    stop = 0
    G_new=G
    H=G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    for i in range((alpha)):
        counter=0
        for nodes in conns:
            if nx.has_path(G,nodes[0],nodes[1]):
                path=nx.dijkstra_path(G,nodes[0],nodes[1],weight='weight')
                path_edges = list(zip(path,path[1:]))
                G_new=remove_edge_path(path_edges,G)
                stop = check_cs(cs,G_new)
                if stop==1:
                    print('stopping in if')
                    return(H,stop)

            else:
                stop = check_cs(cs,G_new)
                if stop==1:
                    print('stopping in else')
                    return(H,stop)

                print('no connection')
                counter+=1
                if counter == len(conns):
                    stop=1
                    return(G,stop)
    return(G,stop)
                    

def find_resistance(G,node):
    L=nx.laplacian_matrix(G)
    M = np.linalg.pinv(L.toarray())
    re=M[node[0],node[0]]+M[node[1],node[1]]-2*M[node[1],node[0]]
    return(re)

def laplacian_resistance_plot(G,coords,pin_indxs,epochs):
    fig,axes = plt.subplots(2,2)
    L=nx.laplacian_matrix(G,nodelist=range(0,len(coords)))
    M = np.linalg.pinv(L.todense())
    xi = np.linspace(0.,1.,100)
    yi = np.linspace(0.,1.,100)
    x=coords[:,0]
    y=coords[:,1]
    fig,axes = plt.subplots(2,2)
    for pins in pin_idxs:
        node=pins
        log=[] 
        for i in range(len(coords)):
            re=M[node,node]+M[i,i]-2*M[i,node]
            log.append(re)
        z=np.asarray(log)
        zi=ml.griddata(x,y,z,xi,yi,interp='linear')
        axes.flat[pins].contourf(xi,yi,zi)
    
    plt.savefig('re_' + str(epochs) + '.png')


def cool_plot(G_temp):
    G=[]
    G=G_temp
    ncenter=5
    p = dict(nx.single_source_shortest_path_length(G, ncenter))
    
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()),
                           node_size=80,
                           node_color=list(p.values()),
                           cmap=plt.cm.Reds_r)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.show()        

#hyperparameters
radii_pin = 0.2
radii_fiber = 0.05
alpha = 4
ratio = .25

#pin locations
pins_x=[0,0,1,1]
pins_y=[1,.25,.75,0]
pin_idxs=[0,1,2,3]


#desired connections
c1 = [0,2]
c2 = [1,3]
cs=np.asarray([c1,c2],np.int32)

#number of fibers
fibers = 2000

#random fibers
x=np.random.rand(fibers)
y=np.random.rand(fibers)
fibers = [x,y]
fibers = np.transpose(np.vstack((x,y)))

#pins and concat with fibers
pins = np.transpose(np.vstack((np.asarray(pins_x),np.asarray(pins_y))))
coords = np.vstack((pins,fibers))

#instantiate and create graph        
G = nx.Graph()
#build out nodes and edges
G=build_graph(G,coords,radii_pin,radii_fiber)
#get the positions
pos = nx.get_node_attributes(G, 'pos')
#get number of edges
N_edges_orig = G.number_of_edges()
N_edges_remaining = round(N_edges_orig*(1-ratio))
#

#initialize for loop
N_edges_new = G.number_of_edges()
loop = tqdm(total=(N_edges_orig-N_edges_remaining), position = 0)
re_log=[]
large_re_log=[]
conns=connections(pin_idxs,cs)
poss_conn_all = np.asarray(list(itertools.combinations(pin_idxs,2)))
i=0
stop=0

#loop while threshold has not been reached (number of edges removed)
while N_edges_new>N_edges_remaining:
    N_edges_old = N_edges_new
    
    #pare the network
    [G_new,stop]=pare_network(conns,G,alpha,cs)
    
    #stop the training if the stop criteria is met
    if stop==1:
        print('stopping')
        cool_plot(G_new)
        break
    
    #find edge removal progress
    N_edges_new = G.number_of_edges()
    delta_edges = N_edges_old-N_edges_new
    loop.update(delta_edges)
    
    #get the shortest connections for all of the pins
    temp=[]
    for j in range(len(poss_conn_all)):
        if nx.has_path(G,poss_conn_all[j,0],poss_conn_all[j,1]):
            temp.append(nx.dijkstra_path_length(G,poss_conn_all[j,0],poss_conn_all[j,1]))
        else:
            temp.append(float("inf"))
    if i==0:
        large_re_log=temp
    else:
        large_re_log=np.vstack((large_re_log,temp))
    i+=1

    #if i%50==0:
    #    laplacian_resistance_plot(G,coords,pin_idxs,i)
#    if i%20==0:
#        re=find_resistance(G,cs[0,:])
#        re_log.append(re)           
        

plt.plot(large_re_log)
plt.legend(('1','2','3','4','5','6'))



laplacian_resistance_plot(G,coords,pin_idxs,i)
