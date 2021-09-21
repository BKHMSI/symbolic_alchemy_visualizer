import numpy as np 
from typing import Dict, List, Tuple

from dm_alchemy import symbolic_alchemy
from dm_alchemy.types.graphs import Graph
from vpython import box, vector, color, arrow, text, cylinder, sphere, button, scene, rate

from constants import *

def draw_edge(a:vector, b:vector) -> box:
    return box(pos=(a+b)/2, length=b.x-a.x+0.1, height=b.y-a.y+0.1, width=b.z-a.z+0.1, opacity=1)

def draw_arrow(a:vector, b:vector, shift:vector, c:color, dir:bool) -> arrow:
    axis, pos = (b-a, a+shift) if dir else (a-b, b-shift)
    return arrow(pos=pos, axis=axis/2, color=c, shaftwidth=0.1, opacity=0.25)

def draw_arrows(potions:symbolic_alchemy.Potion, game_state, state) -> List[arrow]:
    arrows = []
    potion_index_map = [0]*6
    for potion in potions:
        p_color = int(np.round(state[15+2*potion.idx]*3 + 3))
        if p_color >= 6: continue
        dim_index = int(potion.as_index)
        potion_index_map[dim_index] = ARROWS_COLOR_MAP[p_color]

    for pos, colour in enumerate(potion_index_map):
        if colour == 0: continue
        map_idx = pos // 2
        shift = ARROWS_MAP[map_idx][1]
        for arrow in ARROWS_MAP[map_idx][0]:
            node_a = game_state._graph.node_list.get_node_by_idx(arrow[2][0])
            node_b = game_state._graph.node_list.get_node_by_idx(arrow[2][1])
            if game_state._graph.edge_list.has_edge(node_a, node_b):
                arrows += [draw_arrow(vector(*arrow[0]), vector(*arrow[1]), shift=vector(*shift), c=colour, dir=pos%2!=0)]
    return arrows

def draw_cube(coords: List[List[Tuple[int,int,int]]], graph:Graph) -> List[box]:
    edges = []
    for i, (coord_a, coord_b) in enumerate(coords):
        start_node = graph.node_list.get_node_by_idx(NODES_MAP[i][0])
        end_node = graph.node_list.get_node_by_idx(NODES_MAP[i][1])
        if graph.edge_list.has_edge(start_node, end_node):
            edges += [draw_edge(vector(*coord_a), vector(*coord_b))]
    return edges

def draw_potions(state, potions):
    p_pos = -1.35
    for i, idx in enumerate(range(15, 39, 2)):
        potion_color = int(np.round(state[idx]*3 + 3))
        if potions[i] == 0:
            potions[i] = cylinder(pos=vector(p_pos, 2, 0), axis=vector(0,0,0.1), radius=0.1, color=ARROWS_COLOR_MAP[potion_color])
        else:
            potions[i].color = ARROWS_COLOR_MAP[potion_color]
        p_pos += 0.25
    return potions

def draw_stones(stones_state, stones):
    for stone in stones_state:
        if stones[stone.idx] == 0:
            stones[stone.idx] = sphere(pos=vector(*stone.latent)*1.3+vector(0.15,0,0)*stone.idx, radius=0.1, color=STONE_COLORS[stone.idx])
        else:
            stones[stone.idx].pos = vector(*stone.latent)*1.3+vector(0.15,0,0)*stone.idx
    return stones
    
def draw_text_coords():
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                text(text=f"({i},{j},{k})", pos=vector(i,j,k)+vector(i,j,k)*0.2, align="center", height=0.1)

def convert_states(states):
    n_workers = states.shape[0]
    stone_feats = states[:, :15]
    potions = np.zeros((n_workers, 6))
    for idx in range(15, 39, 2):
        potion_colors = np.array(list(map(int, np.round(states[:, idx]*3+3))))
        mask = states[:, idx+1] == 0
        potions[mask, potion_colors[mask]] += 1
    return np.concatenate([stone_feats, potions], axis=-1)

def to_env_actions(states, actions):
    stone_indices = (actions-1) // 7
    potion_color_indices = (actions-1) % 7

    potion_real_indices = np.ones_like(potion_color_indices) * -1
    for i, idx in enumerate(range(15, 39, 2)):
        potion_colors = np.array(list(map(int, np.round(states[:, idx]*3+3))))
        potion_mask = (potion_colors == potion_color_indices) * (states[:, idx+1] == 0)
        potion_real_indices[potion_mask] = i

    env_actions = stone_indices * 13 + 2 + potion_real_indices

    env_actions[potion_real_indices==-1] = 0

    env_actions[actions==0] = 0
    env_actions[actions==7] = 1
    env_actions[actions==14] = 14
    env_actions[actions==21] = 27

    return env_actions