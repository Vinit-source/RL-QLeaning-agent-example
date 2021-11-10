from tkinter import *
import random, time, operator
import numpy as np
from PIL import ImageTk, Image
import logging

logging.basicConfig(filename='logs/reports.log', level=logging.DEBUG, filemode='w', format='\n%(asctime)s\n%(message)s')
logging.Formatter('\n%(asctime)s - %(message)s')

# Define useful parameters
size_of_board = 600
rows = 8
cols = 8
# scale is (x, 7) - (0, y)
DELAY = 100
NEPISODES = 500
symbol_size = (size_of_board / 3 - size_of_board / 8) / 2
symbol_thickness = 2
RED_COLOR = "#EE4035"
BLUE_COLOR = "#0492CF"
Green_color = "#7BC043"

BLUE_COLOR_LIGHT = '#67B0CF'
RED_COLOR_LIGHT = '#EE7E77'
row_h = int(size_of_board / rows)
col_w = int(size_of_board / cols)

class AgentWorld:
    # ------------------------------------------------------------------
    # Initialization Functions:
    # ------------------------------------------------------------------
    def __init__(self):
        self.window = Tk()
        self.window.title("Agent-World")
        self.canvas = Canvas(self.window, width=size_of_board, height=size_of_board)
        self.canvas.pack()
        # Input from user in form of clicks and keyboard
        # self.window.bind("<Key>", self.key_input)
        self.window.bind("<Button-1>", self.mouse_input)
        self.play_again()
        self.toggle = False
    
    
    def initialize_board(self):
        self.board = []
        self.goal_cnd = []
        for i in range(4,8):
            for j in range(0, 4):
                self.goal_cnd.append((i,j))
        self.walls = [(1,5), (2,5), (2,3), (2,2), (2,1), (2,0), (4,6), (4,5), (4,4), (4,3), (4,1), (5,3), (5,1), (6,3), (6,6), (7,6)]
        self.portals = [(7,7), (1,2), (3,1)]
        self.power = (7,3)
        self.goal_cnd = list(set(self.goal_cnd) - set(self.walls) - set(self.portals) - set({(7,3)}))
        self.img = Image.open("images/power.jpg")
        self.powerimg = ImageTk.PhotoImage(self.img)
        self.img = Image.open("images/portal.jpg")
        self.portalimg = ImageTk.PhotoImage(self.img)

        # board variable is initialized
        for i in range(rows):
            for j in range(cols):
                self.board.append((i, j))
        # ------------------------------------------------------------------
        # Initial Board Drawings: (initialize_board)
        # ------------------------------------------------------------------

        # Draw horizontal lines
        for i in range(rows):
            self.canvas.create_line(
                i * size_of_board / rows, 0, i * size_of_board / rows, size_of_board,
            )

        # Draw vertical lines
        for i in range(cols):
            self.canvas.create_line(
                0, i * size_of_board / cols, size_of_board, i * size_of_board / cols,
            )
        
        # Draw walls
        for wall in self.walls:
            x1 = wall[0] * row_h 
            y1 = wall[1] * col_w 
            x2 = x1 + row_h
            y2 = y1 + col_w
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=BLUE_COLOR)
        
        # Draw Power
        self.canvas.create_image(self.power[0] * row_h + 1, self.power[1] * col_w + 1, anchor=NW, image=self.powerimg)

        # Draw portals
        for portal in self.portals:
            x1 = portal[0] * row_h + 1
            y1 = portal[1] * col_w + 1
            self.canvas.create_image(x1, y1, anchor=NW, image=self.portalimg)
        
        # Draw Goal
        self.goal_loc = random.choice(self.goal_cnd)
        # self.goal_loc = (5,5)
        x1, y1 = self.goal_loc
        x1 = x1 * row_h
        y1 = y1 * col_w
        x2 = x1 + row_h
        y2 = y1 + col_w
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=RED_COLOR, tags='goal')
        self.canvas.create_text(x1 + row_h//2, y1 + col_w//2, text='G', tags='goal')
        # print(self.toggle)

        # ------------------------------------------------------------------
        # More Initializations : (initialize_board)
        # ------------------------------------------------------------------

        # Define state space
        self.Ny = cols  # y grid size
        self.Nx = rows  # x grid size
        self.state_dim = (cols, rows)   
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"Up": 0, "Right": 1, "Down": 2, "Left": 3}
        self.action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # translations
        # Define rewards table
        self.R = self._build_rewards(self.goal_loc)  # R(s,a) agent rewards
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset_agent(self):
        # Reset game state
        self.agent = (0, 7)
        self.reached_goal = False

        # Draw agent
        x1, y1 = self.agent
        x1 = x1 * row_h
        y1 = y1 * col_w
        x2 = x1 + row_h
        y2 = y1 + col_w
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=Green_color, tags='agent')
        self.canvas.create_text(x1 + row_h//2, y1 + col_w//2, text='A', tags='agent')
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        x, y = self.agent[0], self.agent[1]
        if (y > 0 and (x, y-1) not in self.walls):  # no passing top-boundary and walls
            actions_allowed.append(self.action_dict["Up"])
        if (y < self.Ny - 1 and (x, y+1) not in self.walls):  # no passing bottom-boundary and walls
            actions_allowed.append(self.action_dict["Down"])
        if (x > 0 and (x-1, y) not in self.walls):  # no passing left-boundary and walls
            actions_allowed.append(self.action_dict["Left"])
        if (x < self.Nx - 1 and (x+1, y) not in self.walls):  # no passing right-boundary and walls
            actions_allowed.append(self.action_dict["Right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed
    
    def step(self, action):
        # Collect reward
        reward = self.R[self.agent + (action,)]
        if self.agent in self.portals:
            # Teleport to Power
            state_next = self.power
            # Update agent
            self.agent = state_next
            return state_next, reward, False
        else:
            # Evolve agent state
            state_next = (self.agent[0] + self.action_coords[action][0],
                        self.agent[1] + self.action_coords[action][1])

            # Terminate if we reach bottom-right grid corner
            done = (state_next == self.goal_loc) 
            # Update state
            self.agent = state_next
            return state_next, reward, done
    
    # ------------------------------------------------------------------
    # Drawing Functions:
    # The modules required to draw required game based object on canvas
    # ------------------------------------------------------------------

    def play_again(self):
        self.canvas.delete("all")
        self.initialize_board()
        self.reset_agent()
        self.begin_time = time.time()

    def update_step(self, agent):
        action = agent.get_action(self)  # get action
        state_next, reward, done = self.step(action)  # evolve state by action
        # self.agent = state_next # resume here
        # print(f'state_next: {state_next}')
        agent.train((self.agent, action, state_next, reward, done))  # train agent
        # Update Agent location on grid
        self.canvas.delete('agent')
        x1, y1 = state_next
        x1 = x1 * row_h
        y1 = y1 * col_w
        x2 = x1 + row_h
        y2 = y1 + col_w
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=Green_color, tags='agent')
        self.canvas.create_text(x1 + row_h//2, y1 + col_w//2, text='A', tags='agent')
        if done:
            self.reached_goal = True
        return state_next, reward, done

    def mainloop(self, agent):
        print("\nTraining agent...\n")
        LOAD = False
        SAVE = True
        if LOAD:
            agent.Q = np.load('saved_model/kbase.npy')
            logging.debug(f'Knowledge Base: {agent.Q}')
            agent.display_result(self)
        optimal_paths = {}
        cost_optimal = {}
        for episode in range(NEPISODES):
            # Generate an episode
            iter_episode, reward_episode = 0, 0
            path = []
            self.play_again()  # starting state
            while True:  
                self.window.update()
                state_next, reward, done = self.update_step(agent)
                # self.window.after(DELAY)  # Unccomment to make the transitions slow
                iter_episode += 1
                reward_episode += reward
                path.append((state_next[0], 7 - state_next[1]))
                if done:
                    break
                # state = state_next  # transition to next state

            # Decay agent exploration parameter
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.636)

            if not self.goal_loc in optimal_paths:
                optimal_paths[self.goal_loc] = path
                cost_optimal[self.goal_loc] = reward_episode
            else:
                if reward_episode > cost_optimal[self.goal_loc]:
                    optimal_paths[self.goal_loc] = path
                    cost_optimal[self.goal_loc] = reward_episode
            # Log reports
            logging.debug('='*30+f" episode {episode+1}/{NEPISODES} "+'='*30+f'\neps = {agent.epsilon:.3F}\n \
            path followed = {path}\npath_cost = {reward_episode:.1F}\n optimal_path = {optimal_paths[self.goal_loc]}\n \
                optimal cost = {cost_optimal[self.goal_loc]}\nKnowledge base: {agent.Q}')

            # Print
            if (episode == 0) or (episode + 1) % 10 == 0:
                print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}".format(
                    episode + 1, NEPISODES, agent.epsilon, iter_episode, reward_episode))


            # Print greedy policy
            if (episode == NEPISODES - 1):
                for (key, val) in sorted(self.action_dict.items(), key=operator.itemgetter(1)):
                    print(" action['{}'] = {}".format(key, val))
                maxKey = list(optimal_paths.keys())[np.argmax(list(cost_optimal.values()))]
                logging.debug(f'Best optimal path: {optimal_paths[maxKey]} has optimal_cost: {cost_optimal[maxKey]}')
                if SAVE:
                    np.save('saved_model/kbase.npy', agent.Q, allow_pickle=True)
                agent.display_result(self)
                print()

    def mouse_input(self, event):
        # self.play_again()
        self.toggle = not self.toggle
 
    # ------------------------------------------------------------------
    # Logical Functions:
    # The modules required to carry out game logic
    # ------------------------------------------------------------------

    def _build_rewards(self, goal_loc):
        # Define agent rewards R[s,a]
        r_goal = 100  # reward for arriving at terminal state (bottom-right corner)
        r_nongoal = -0.1  # penalty for not reaching terminal state
        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a] Initialization
        gx, gy = goal_loc
        if gy > 0:
            R[gx, gy - 1, self.action_dict["Down"]] = r_goal  # arrive from above
        if gx > 0:
            R[gx - 1, gy, self.action_dict["Right"]] = r_goal  # arrive from the right
        if gx < self.Nx - 1:
            R[gx + 1, gy, self.action_dict["Left"]] = r_goal  # arrive from the left
        if gy < self.Ny - 1:
            R[gx, gy + 1, self.action_dict["Up"]] = r_goal  # arrive from the below
        return R

class Agent:
    def __init__(self, env):
        # Store state and action dimension 
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.alpha = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.agent
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], 7 - state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + alpha * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  alpha = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, _) = memory
        sa = state + (action,)
        self.Q[sa] += self.alpha * (reward + self.gamma*np.max(self.Q[state_next]) - self.Q[sa])

    def display_result(self, env):
        # Clear the game scene
        env.canvas.delete("all")
        # Fetch the preferred directions in the final episode
        greedy_policy = -1 * np.ones((self.state_dim[0], self.state_dim[1]), dtype=int)
        for x in range(self.state_dim[0]):
            for y in range(self.state_dim[1]):
                if (x, y) not in env.walls:
                    if self.Q[x, y, :].any():
                        Qi = np.flatnonzero(self.Q[x, y, :])
                        temp = np.full_like(self.Q[x, y, :], -np.inf)
                        temp[Qi] = self.Q[x, y, :][Qi]
                        greedy_policy[x, y] = np.argmax(temp)
                        # Visualization of preferred direction in the final episode
                        x1, y1 = x, y
                        x1 = x1 * row_h
                        y1 = y1 * col_w
                        x2 = x1 + row_h
                        y2 = y1 + col_w
                        env.canvas.create_rectangle(x1, y1, x2, y2, fill=Green_color, tags='final_op')
                        env.canvas.create_text(x1 + row_h//2, y1 + col_w//2, text=f'{list(env.action_dict.keys())[greedy_policy[x, y]]}', tags='final_op')
            
        print("\nGreedy policy(y, x):")
        print(np.transpose(greedy_policy))
        print()
        while(True):
            if not env.toggle:
                env.window.update()




if __name__ == '__main__':
    game_instance = AgentWorld()
    agent = Agent(game_instance)
    game_instance.mainloop(agent)
