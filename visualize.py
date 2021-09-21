import os
import yaml
import argparse
import numpy as np

from typing import Dict

import torch
from torch.nn import functional as F
from vpython import rate
from vpython.vpython import wtext

from dm_alchemy import symbolic_alchemy
from dm_alchemy.encode import chemistries_proto_conversion
from dm_alchemy.types import utils

from agents.a2c_epn import A2C_EPN
from agents.components.episodic_memory import Episodic_Memory

from utils import *

class Visualizer:
    def __init__(self, config: Dict) -> None:
        self.chems = chemistries_proto_conversion.load_chemistries_and_items('chemistries/perceptual_mapping_randomized_with_random_bottleneck/chemistries')
        self.agent = A2C_EPN(config["agent"], config["task"]["n-actions"])
        self.device = config["device"]
        self.config = config
        self.dt = config["dt"]
        
        self.agent.to(config["device"])
        self.agent.eval()

        filepath = config["load-path"]
        print(f"> Loading Checkpoint {filepath}")
        model_data = torch.load(filepath, map_location=torch.device(self.device))
        self.agent.load_state_dict(model_data["state_dict"])


    def setup(self, episode: int) -> None:

        chem, items = self.chems[episode]
        level_name = 'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck'
        self.env = symbolic_alchemy.get_symbolic_alchemy_fixed(chemistry=chem, episode_items=items, see_chemistries={
            'input_chem': utils.ChemistrySeen(content=utils.ElementContent.BELIEF_STATE, precomputed=level_name)
        }, max_steps_per_trial=15)

        self.n_actions = self.config["task"]["n-actions"]
        self.n_potions = self.config["task"]["n-potions"]

        timestep = self.env.reset()
        self.state = timestep.observation["symbolic_obs"]
        self.p_action = np.zeros((1, self.n_actions))
        self.p_reward = np.zeros((1, 1))

        self.ht = torch.zeros(1, self.agent.hidden_dim).float().to(self.device)
        self.ct = torch.zeros(1, self.agent.hidden_dim).float().to(self.device)

        potions = self.env.game_state.existing_potions()
        stones = self.env.game_state.existing_stones()

        self.edges = draw_cube(COORDS, self.env.game_state._graph)
        draw_text_coords()
        self.arrow_objects = draw_arrows(potions, self.env.game_state, self.state)

        self.arrow_beliefs = [0]*3

        potion_objects = [0]*12
        self.potion_objects = draw_potions(self.state, potion_objects)

        stone_objects = [0]*3
        self.stone_objects = draw_stones(stones, stone_objects) 

        self.episode_reward = 0
        button(text="Step", pos=scene.title_anchor, bind=self.step)
        self.reward_text = wtext(text=f"<b>Episode Reward: {self.episode_reward}</b>", pos=scene.title_anchor)
        self.trial_step_text = wtext(text=f" | <b>Trial: 1</b> | <b>Step: 0</b>", pos=scene.title_anchor)
        self.action_text = wtext(text=f" | <b>Action: -</b> | <b>Stone: -</b>", pos=scene.title_anchor)

    def step(self, dummy) -> None:
        if not self.running: return

        mem_mask = self.agent.memory.generate_mask()
        states = np.array([self.state])
        model_states = convert_states(states)
        logit, value, (self.ht, self.ct) = self.agent((
            torch.from_numpy(model_states).float().to(self.device),
            torch.from_numpy(self.p_action).float().to(self.device),
            torch.from_numpy(self.p_reward).float().to(self.device),
            torch.from_numpy(self.agent.memory.memory).float().to(self.device),
            torch.from_numpy(mem_mask).float().to(self.device),
            (self.ht, self.ct)
        ))

        prob = F.softmax(logit, dim=-1)
        action = np.array([prob.argmax().item()])

        env_actions = to_env_actions(states, action)

        timestep = self.env.step(env_actions[0])
        done = timestep.last()
        next_state = timestep.observation["symbolic_obs"]

        self.episode_reward += timestep.reward
        self.reward_text.text = f"<b>Episode Reward: {int(self.episode_reward)}</b>"
        self.trial_step_text.text = f" | <b>Trial: {self.env.trial_number+1}</b> | <b>Step: {self.env._steps_this_trial}</b>"
        
        for i, idx in enumerate(range(15, 39, 2)):
            if next_state[idx+1] == 1:
                self.potion_objects[i].visible = False

        self.stone_objects = draw_stones(self.env.game_state.existing_stones(), self.stone_objects)

        if self.env.is_new_trial(): 
            
            self.trial_step_text.text = f" | <b>Trial: {self.env.trial_number+1}</b> | <b>Step: {self.env._steps_this_trial}</b>"
            self.action_text.text = f" | <b>Action: -</b> | <b>Stone: -</b>"

            self.state = next_state
            for arrow in self.arrow_objects: arrow.visible = False  
            self.arrow_objects = draw_arrows(self.env.game_state.existing_potions(), self.env.game_state, self.state)

            self.potion_objects = draw_potions(self.state, self.potion_objects)
            for potion in self.potion_objects: potion.visible = True 
            self.stone_objects = draw_stones(self.env.game_state.existing_stones(), self.stone_objects)
            for stone in self.stone_objects: stone.visible = True 
        
            self.p_action = np.zeros((1, self.n_actions))
            self.p_reward = np.zeros((1, 1))
        elif not done:

            stone_idx = (int(action)-1) // 7
            potion_color_idx = (int(action)-1) % 7

            if action == 0:
                stone_idx = 0
                potion_color_idx = 7
            
            self.action_text.text = f" | <b>Action: {POTION_COLOURS[potion_color_idx]}</b> | <b>Stone: {STONE_MAP[stone_idx]}</b>"
            if potion_color_idx == 6:
                self.stone_objects[stone_idx].visible = False

            stone_feats = self.state[stone_idx*5:(stone_idx+1)*5]
            stone_feats_p1 = next_state[stone_idx*5:(stone_idx+1)*5]

            if potion_color_idx < 6 and any(stone_feats!=stone_feats_p1):
                self.arrow_beliefs[potion_color_idx//2] = 1

            self.agent.memory.push(np.array([
                stone_idx,
                *stone_feats,
                *np.eye(self.n_potions)[potion_color_idx],
                *stone_feats_p1,
            ]))

            penalty = 0
            if len(self.env.game_state.existing_stones()) > 0:
                # if action doesn't have any effect on stone and action is not NoOp
                if all(self.state[:15]==next_state[:15]) and int(action) != 0:
                    penalty = -0.2
                # choosing an empty or non-existent potion or using a cached stone
                elif all(self.state==next_state) and int(action) != 0:
                    penalty = -1
                
                # choosing the same potion color consecutively 
                if int(action) == np.array(self.p_action).argmax() and int(action) % 7 != 0:
                    penalty += -1

            self.p_action = np.array([np.eye(self.n_actions)[int(action)]])
            self.p_reward = np.array([[timestep.reward + penalty]])
            self.state = next_state
        else:
            self.running = False

        for arrow in self.arrow_objects: arrow.visible = False 
        self.arrow_objects = draw_arrows(self.env.game_state.existing_potions(), self.env.game_state, self.state)
        for arrow in self.arrow_objects:
            index = ARROWS_COLOR_MAP.index(arrow.color)
            if self.arrow_beliefs[index//2] == 1:
                arrow.opacity = 1

    def animate(self) -> None:
        self.running = True
        while True:
            rate(1/self.dt)
            if not self.running:
                break
            
if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    visualizer = Visualizer(config)
    visualizer.setup(episode=2)
    visualizer.animate() 
