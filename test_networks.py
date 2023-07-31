import numpy as np
import math
from numpy.linalg import inv as inv_mat
import copy as cp
from sympy.matrices import Matrix, zeros
from math import acos, asin, atan, sqrt, cos, sin, tan
from networks import *

np.set_printoptions(precision=5, suppress=True)


def printTitle(str):
    l=50
    str='Verify '+str
    print('='*l+"\n="+' '*(l//2-1-len(str)//2)+str+' '*(l//2-1-len(str)+len(str)//2)+'=\n'+'='*l)

def verify(network, title=None, wholecompute=True):
    if title is not None:
        printTitle(title)
    subnetwork=networksWithPartition(graph=network['subNetwork'])
    subnetwork.compute_DN_map()
    eig=np.linalg.eigh(subnetwork.DN_map)[0]
    print(f"The eigenvalues of subnetwork are given by {eig}.")
    print(f"It has index {len([y for y in eig if y<-1e-6])}, nullity {len([y for y in eig if -1e-6<y and y<1e-6])}.")
    subnetwork.compute_boundary_DN()
    print(f"The Dirichlet-to-Neumann map of subnetwork on boundary with respect to the order of point list {[p for _,p in subnetwork.bdy_DN_point_list]} is given by\n {subnetwork.bdy_DN_map}")
    Ni=subnetwork.convert2nets()
    if wholecompute:
        print('-'*50+'\nNow, let us compute the Dirichlet-to-Neumann map of the network with partition.')
        N_Dict={
            'N':[cp.deepcopy(Ni) for _ in range(len(network['NetworkInteriorEndPoint']))],
            'renameEndPoint':network['NetworkInteriorEndPoint']
        }
        whole_network=networksWithPartition(N_dict=N_Dict)
        whole_network.compute_DN_map()
        eig=np.linalg.eigh(whole_network.DN_map)[0]
        print(f'The eigenvalues of the Dirichlet-to-Neumann map with respect to this partition are given by\n{eig}')
        print(f'It has {len([y for y in eig if y>1e-6])} positive eigenvalues.')
        print(f"It has index {len([y for y in eig if y<-1e-6])}, nullity {len([y for y in eig if -1e-6<y and y<1e-6])}.")
        

##=======================##
##  Tetrahedron          ##
##=======================##

verify({
        "subNetwork":{
            math.acos(-1.0/3)/2: [(1,4),(2,4),(3,4)]
        },
        "NetworkInteriorEndPoint":[[1,2,3],[1,4,6],[3,5,6],[2,4,5]]
    },'tetrahedron')

##=======================##
##  Cube                 ##
##=======================##

verify({
        "subNetwork":{
            math.acos(1.0/3): [(1,4),(2,4),(3,4)]
        },
        "NetworkInteriorEndPoint":[[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
    },'cube')


##=======================##
##  Dodecahedron         ##
##=======================##

verify({
        "subNetwork":{
            math.acos(sqrt(5)/3): [(7,6,8),(1,6)],
            math.acos(sqrt(5)/3)/2: [(2,7),(3,7),(4,8),(5,8)],
        },
        "NetworkInteriorEndPoint":[[1,3,6,5,2],[1,4,8,7,3],[1,2,10,9,4],[14,13,10,5,11],[14,11,6,7,12],[14,12,8,9,13]]
    },'dodecahedron')
##=======================##
## Triangle prism        ##
##=======================##
printTitle('triangle prism')
print('We compute the index and nullity of N1 here. According to our paper, it should has index 1 and nullity 2.')
l1 = math.acos(-1/3)
l2 = math.acos(7/9)

verify({
    "subNetwork":{
        l1/2:[('a','o'),('b','o')],
        l2/2:[('c','o')]
    },
    "NetworkInteriorEndPoint":[['a1','a2','c1'],['a2','a3','c2'],['a3','a1','c3']]
})


##=======================##
## Pentagon prism        ##
##=======================##

printTitle('pentagon prism')
print('We compute the index and nullity of N1 here. According to our paper, it should has index 1 and nullity 2.')
l1 = math.acos(math.sqrt(5.0)/3)
l2 = math.acos((3.0-5.0*math.sqrt(5.0)/3)/(5-math.sqrt(5.0)))

verify({
    "subNetwork":{
        l1/2:[('a','o'),('b','o')],
        l2/2:[('c','o')]
    },
    "NetworkInteriorEndPoint":[['a1','a2','c1'],['a2','a3','c2'],['a3','a4','c3'],['a4','a5','c4'],['a5','a1','c5']]
})

##=======================##
##  4-4 type             ##
##=======================##

# These numbers are taken from J. Taylor's paper

l1=83.80167087/180*math.pi
l2=58.25684287/180*math.pi
l3=13.55944752/180*math.pi

# lables of endpoints are taken from our paper
graph = {
        l1/2:[('P_1','P')],
        l2:[('P\'','P')],
        l3/2:[('Q_1','P')]
    }

printTitle('4-4 type')
print('We focus on the network N111 at first.')
verify({"subNetwork":graph},None,False
)

print("-"*50+"\nNow, let us compute the Dirichlet-to-Neumann map of N11 at P'.")
subnetwork=networksWithPartition(graph=graph)
subnetwork.compute_boundary_DN()
Ni=subnetwork.convert2nets()
N_Dict={
    'N':[cp.deepcopy(Ni),cp.deepcopy(Ni),network(l1/2)],
    'renameEndPoint':[['P_1','P\'','Q_1'],['P_3','P\'','Q_2'],['P_2','P\'']]
}
whole_network=networksWithPartition(N_dict=N_Dict)
whole_network.compute_DN_map()
eig=np.linalg.eigh(whole_network.DN_map)[0]
print(f'The eigenvalues of the Dirichlet-to-Neumann map with respect to this partition are given by\n{eig}')
print(f'It has {len([y for y in eig if y>1e-6])} positive eigenvalues.')
print(f"It has index {len([y for y in eig if y<-1e-6])}, nullity {len([y for y in eig if -1e-6<y and y<1e-6])}.")
print("-"*50+"\nNow, let us compute the Dirichlet-to-Neumann map of N1 acts on [1,-1,0]'.")
whole_network.compute_boundary_DN()
res=(whole_network.bdy_DN_map[np.ix_([0,2,4],[0,2,4])] @ np.array([[1,-1,0]]).T).T[0]
print(res)
print(f'So the lambda value is {res[0]}, This is a positive number.')

subnetwork=networksWithPartition(graph=graph)
Ni=whole_network.convert2nets()
whole_network=networksWithPartition(N_dict={
    "N":[cp.deepcopy(Ni),cp.deepcopy(Ni)],
    "renameEndPoint":[[1,4,3,5,2],[1,7,3,6,2]]
})
whole_network.compute_DN_map()
eig=np.linalg.eigh(whole_network.DN_map)[0]
print(f'The eigenvalues of the Dirichlet-to-Neumann map with respect to this partition are given by\n{eig}')
print(f'It has {len([y for y in eig if y>1e-6])} positive eigenvalues.')
print(f"It has index {len([y for y in eig if y<-1e-6])}, nullity {len([y for y in eig if -1e-6<y and y<1e-6])}.")

print("-"*50+"\nNow, let us compute the Dirichlet-to-Neumann map of N.")

Ni=whole_network.convert2nets()
whole_network=networksWithPartition(N_dict={
    'N':[cp.deepcopy(Ni) for _ in range(2)],
    'renameEndPoint':[[1,2,3,4],[3,1,4,2]]
})
whole_network.compute_DN_map()
eig=np.linalg.eigh(whole_network.DN_map)[0]
print(f'The eigenvalues of the Dirichlet-to-Neumann map with respect to this partition are given by\n{eig}')
print(f'It has {len([y for y in eig if y>1e-6])} positive eigenvalues.')
print(f"It has index {len([y for y in eig if y<-1e-6])}, nullity {len([y for y in eig if -1e-6<y and y<1e-6])}.")

##=======================##
##  6-3 type             ##
##=======================##

printTitle('6-3 type')
l1=2*asin(1/sqrt(3))
l3=2*asin((sqrt(3)-sqrt(2))/(2*sqrt(3)))
# 
graph1={
        l1/2:[(1,4),(2,4),(3,4)],
    }
graph2={
        l1/2:[(1,4),(2,4)],
        l3/2:[(3,4)]
    }
# 
print('-'*50+'\nCompute the Dirichlet-to-Neumann map of N11')
verify({
    "subNetwork":graph1,
    "NetworkInteriorEndPoint":[[1,2,3,4],[2,3,4,1]]
},title=None,wholecompute=False)
# 
print('-'*50+'\nCompute the Dirichlet-to-Neumann map of N12')
verify({
    "subNetwork":graph2,
    "NetworkInteriorEndPoint":[[1,2,3,4],[2,3,4,1]]
},title=None,wholecompute=False)
# 
N1=networksWithPartition(graph=graph1).convert2nets()
N2=networksWithPartition(graph=graph2).convert2nets()
N_Dict={
    'N':[cp.deepcopy(N1),cp.deepcopy(N2),cp.deepcopy(N1),cp.deepcopy(N2)],
    'renameEndPoint':[[1,2,5],[2,3,6],[3,4,7],[4,1,8]]
}
# 
print('-'*50+'\nNow, let us compute the Dirichlet-to-Neumann map of the network with partition.')
whole_network=networksWithPartition(N_dict=N_Dict)
whole_network.compute_DN_map()
eig=np.linalg.eigh(whole_network.DN_map)[0]
print('The Dirichlet-to_Neumann map of N1 with respect to its partition is given by')
print(whole_network.DN_map)
print(f'The eigenvalues of the Dirichlet-to-Neumann map with respect to this partition are given by\n{eig}')
print(f'It has {len([y for y in eig if y>1e-6])} positive eigenvalues.')
print(f"It has index {len([y for y in eig if y<-1e-6])}, nullity {len([y for y in eig if -1e-6<y and y<1e-6])}.")

##=======================##
##  8-2 type             ##
##=======================##

l_straight = 2*asin((((2**(1/4)-1)**2/6)+(2-sqrt(2))**2/12)**(1/2))
l_connection_arc = 2*asin(((2-sqrt(2))/3)**(1/2))
l_square_arc = acos(1/3)

verify({
        "subNetwork":{
            l_square_arc/2:[(1,5),(2,5)],
            l_connection_arc/2:[(3,6),(4,6)],
            l_straight:[(5,6)]
        },
        "NetworkInteriorEndPoint":[[1,2,6,5],[2,3,8,7],[3,4,10,9],[4,1,12,11],[14,13,6,7],[15,14,8,9],[16,15,10,11],[13,16,12,5]]
    },'8-2 type')
