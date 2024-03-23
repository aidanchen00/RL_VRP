#import needed libraries
import gym
from gym import spaces
import numpy as np
import copy
import numpy as np
import math as m
import random as r

class vehicleRoutingEnv(gym.Env):
    def __init__(self, current_customer_list, current_truck_capacity, total_truck_capacity, depot_location):
        #current_customer_list contains a list of lists where each list stores the mapped customer's location and demand and distance of this customer to depot and distance of customer to current location (this changes every state) and position of customer relative to depot and position of customer relative to current position (this changes every state)
        #Each list of customer information will look like [(x,y),a,b,c,d,e]
        #Where x, y are the coordinates, a is the customer demand, b is customer distance to depot, c is customer distance to current truck location, d is position of customer relative to depot, and e is position of customer relative to current poisiton
        #c,e will be recalculated after every action of selecting a customer because your current position will have changed
        #a,b,d can be defined and calculated at the start or environment reset of the problem instance
        #d and e are in coordinate pairs and will look like (x,y)
        #The first list in the list will be for customer m where K represents a customer
        #Customer m is the current customer you are on
        #n is the total number of customers
        #The list for the customer at the end of the customer list will be the customer to the "left" or right before the current customer
        #Therefore the list will look like Km, Km+1, Km+2,Km+3,...,kn,k1,k2,k3
        #When defined depot will be first item in list formatted (x,y)
        #Keeps depot index stored
        self.current_customer_list=current_customer_list
        #current_truck_capacity stores the current truck capacity
        self.current_truck_capacity=current_truck_capacity
        #total_truck_capacity stores the capacity of a fully refilled truck
        self.total_truck_capacity=total_truck_capacity
        #Create start information for reset function to reference
        self.start_current_customer_list=copy.deepcopy(self.current_customer_list)
        self.start_current_truck_capacity=copy.deepcopy(self.current_truck_capacity)
        self.start_total_truck_capacity=copy.depcopy(self.total_truck_capacity)
        self.depot_location=depot_location
        self.depot_index=self.current_customer_list.index(self.depot_location)
        #Define unvisited and visited vector to keep track
        #Names of indexes that are referenced frequently
        #Index values for current customer list information
        self.index_coord=0
        self.index_demand=1
        self.index_dist_to_depot=2
        self.index_dist_to_truck=3
        self.index_pos_rel_depot=4
        self.index_pos_rel_truck=5
        #Action values, 0 for left, 1 for right, 2 to select, 3 to return to depot
        self.action_left=0
        self.action_right=1
        self.action_pickup=2
        self.action_goback=3
        #Light penalty for swiping left and right
        self.left_right_reward = -0.001
        #Defining visited and unvisited list
        self.visited=[]
        self.unvisited=[i for i in range(1,len(self.current_customer_list)+1)]
        
        # num customer plust depot possible actions: 0=Left, 1=right, 2=select, 3=return to depot
        self.action_space = spaces.Discrete(4)  

        # Observation space is a list of floats
        # Observation contains:
        # each customer sublist and its information
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(len(self.unvisited)*9+2,), dtype=np.float32)

    def reorder_customers(self):
        #Suppose you had depot coordinate
        depot=self.depot_location
        #rel_dist list will be a parallel list to reoredered customer location list that stores distances to depot for the customer
        self.rel_dist = np.array([])
        self.reordered_customer_list=[]
        for customer in self.current_customer_list:
            #temporarily soley base ordering off distance. In order to make it generalizable later on consider angle to depot or demand or other attributes
            #Only if not depot, calculate distance. At the end of reorder_customers depot will be inserted to the reordered list
            if customer!=self.depot:
                dist = (m.sqrt(((depot[0]-customer[self.index_coord][0])**2)+(depot[1]-customer[self.index_coord][1])**2))
                np.append((self.rel_dist, (dist,customer))
        self.rel_dist.sort(key=lambda x: x[0])
        for i in range(len(self.rel_dist)):
            self.reordered_customer_list.append(self.rel_dist[i][1])
        self.reordered_customer_list.insert(0,self.depot_location)
        self.current_customer_list=copy.deepcopy(self.reordered_customer_list)
        """"
            #if the list is empty, then the first customer is automatically appended
            if len(self.reordered_customer_list)==0:
                rel_dist.append(dist)
                self.reordered_customer_list.append(customer)
            else:
                for idx in len(range(self.reordered_customer_list)):
                    #if the customer's distance to the depot is smaller than the customer at that index, the customer it inserted before
                    if dist<rel_dist[idx]:
                        rel_dist.insert(idx, dist)
                        self.reordered_customer_list.insert(idx, customer)
                        break
                #if the customer was not inserted at all, it means its distance to the depot is bigger than every customer in the list and therefore, must be appended to the end of the list
                if dist in rel_dist==False:
                    rel_dist.append(idx, dist)
                    self.reordered_customer_list.append(customer)
            """
        #Sets main list to reordered list

            
    #defining reset function for when environment needs to be reset after termination state
    def reset(self):
        self.current_customer_list = copy.deepcopy(self.start_current_customer_list)
        self.current_truck_capacity=copy.deepcopy(self.start_current_truck_capacity)
        self.total_truck_capacity=copy.deepcopy(self.start_total_truck_capacity)
        self.depot_location=self.current_customer_list.index((self.depot_location))
        self.unvisited=[i for i in range(1,len(self.customer_location_list)+1)]
        self.visited=[]
        state = []
        for sublist in self.current_customer_list:
            state.extend([sublist[0][0],sublist[0][1],sublist[1],sublist[2],sublist[3],sublist[4][0],sublist[4][1],sublist[5][0],sublist[5][1]])
        f_state = self.flat_list(state)
        return f_state, {"cust list":self.current_customer_list, "truck cap":self.current_truck_capacity, "max cap":self.total_truck_capacity}

    #defining update function for step, to recalculate values for each customer
    def intialize_attributes(self):
        #Suppose you had depot coordinate
        depot=self.depot_location
        #start lists used as reference when reset is called
        for customer in self.current_customer_list:
            #distance to current truck location
            current_index=self.reordered_customer_list.index(customer)
            customer[self.index_dist_to_truck]=(m.sqrt((\
        (customer[self.index_coord][0]-depot[0])**2)\
        +(customer[self.index_coord][1]-depot[1])**2))
            #position relative to depot
            customer[self.index_pos_rel_depot]=(customer[self.index][0]-depot[0],customer[self.index_coord][1]-depot[1])
            #position relative to current location
            customer[self.index_pos_rel_truck]=(customer[self.index_coord][0]-self.current_customer_list[0][self.index_coord][0],customer[self.index_coord][1]-self.current_customer_list[0][self.index_coord][1])
            self.reordered_customer_list[current_index]=customer
        state = []
        #Create flat state which are the inputs the neural network takes in
        #The neural network takes 
        for sublist in self.current_customer_list:
            state.extend([sublist[self.index_coord][0],sublist[self.index_coord][1],sublist[self.index_demand],sublist[self.index_dist_to_depot],sublist[self.index_dist_to_truck],sublist[self.index_pos_rel_depot][0],sublist[self.index_pos_rel_depot][1],sublist[self.index_pos_rel_truck][0],sublist[self.index_pos_rel_truck][1]])
        f_state = self.flat_list(state)
        return f_state, {"cust list":self.current_customer_list, "truck cap":self.current_truck_capacity, "max cap":self.total_truck_capacity}

    def update(self):
        #Suppose you had depot coordinate
        depot=self.depot_location
        #start lists used as reference when reset is called
        for customer in self.current_customer_list:
            #distance to current truck location
            current_index=self.reordered_customer_list.index(customer)
            customer[self.index_dist_to_truck]=(m.sqrt((\
        (customer[self.index_coord][0]-depot[0])**2)\
        +(customer[self.index_coord][1]-depot[1])**2))
            #position relative to current location
            customer[self.index_pos_rel_truck]=(customer[self.index_coord][0]-self.current_customer_list[0][self.index_coord][0],customer[self.index_coord][1]-self.current_customer_list[0][self.index_coord][1])
            self.reordered_customer_list[current_index]=customer
        state = []
        #Create flat state which are the inputs the neural network takes in
        #The neural network takes in the information for every customer
        for sublist in self.current_customer_list:
            state.extend([sublist[self.index_coord][0],sublist[self.index_coord][1],sublist[self.index_demand],sublist[self.index_dist_to_depot],sublist[self.index_dist_to_truck],sublist[self.index_pos_rel_depot][0],sublist[self.index_pos_rel_depot][1],sublist[self.index_pos_rel_truck][0],sublist[self.index_pos_rel_truck][1]])
        f_state = self.flat_list(state)
        return f_state, {"cust list":self.current_customer_list, "truck cap":self.current_truck_capacity, "max cap":self.total_truck_capacity}
        
    #Checks four options and retunrs mask
    # num customer plust depot possible actions: 0=Left, 1=right, 2=select, 3=return to depot
    #When mask=1, index is valid. When mask=0, index is invalid at that location
    def get_mask(self):
        left  = 0
        right = 0
        pickup= 0
        #If already at depot, then going back to depot is invalid, otherwise depot is always available
        if self.current_customer_list[0]==self.depot_location:
            goback=0
        else:
            goback= 1
        #By default depot is only option
        current_index=self.reordered_customer_list.index(self.current_customer_list[0])
        for customer_index in range(len(self.reordered_customer_list)):
            #If valid positon at all in any customer, then left and right become available
            if self.reordered_customer_list[customer_index][1]<self.current_truck_capacity and customer_index in self.unvisited:
                left = 1
                right= 1
                break
        #If valid current location pickup is valid
        if self.is_valid_position(current_index):
            pickup = 1
        mask = [left, right, pickup, goback]
        return mask

        
    #determining if location inputted is valid
    def is_valid_position(self, index):
        valid=False
        #If you are at depot, you are at valid location
        if index == 0:
            valid=True
        elif index in self.unvisited:
            if self.reordered_customer_list[index][self.index_demand]<=self.current_truck_capacity:
                valid=True
            else:
                valid=False
        return valid
        
    #defining helper functions
    def get_visited(self):
        return self.visited

    def get_unvisited(self):
        return self.unvisited
        
    def step(self, action):
        action = int(action)
        #Current customer index in context of original reordered list
        # num customer plust depot possible actions: 0=Left, 1=right, 2=select, 3=return to depot
        current_index=self.reordered_customer_list.index(self.current_customer_list[0])
        if action==self.action_left:
            #Shifts the list
            self.current_customer_list = self.current_customer_list[len(self.current_customer_list)-1] + self.current_customer_list[0:len(self.current_customer_list)-1]
            #Slight negative reward for taking an action without making a move
            self.depot_index=self.current_customer_list.index(self.depot_location)
            reward=self.left_right_reward
        elif action==self.action_right:
            #Shifts the list
            self.current_customer_list = self.current_customer_list[1:] + self.current_customer_list[0]
            #Sligh negative reward for taking an action without making a move
            self.depot_index=self.current_customer_list.index(self.depot_location)
            reward=self.left_right_reward
        elif action==self.action_pickup:
            if self.is_valid_position(current_index):
                #Adds to visited the current customer's original index, meaning in the context of the original reordered list
                self.visited.append(current_index)
                self.unvisited.remove(current_index)
                #Adjust truck capacity accordingly
                #Adjust customer demand accordingly
                self.current_truck_capacity=self.current_truck_capacity-self.current_customer_list[0][1]
                #Adjust the customer's demand to 0 because the customers order has been filled
                self.current_customer_list[0][self.index_demand]=0
                #Reward is negative distance from current location to old location
                reward=self.current_customer_list[0][self.index_dist_to_truck]
                self.update()
            else:
                # action==self.action_pickup and !self.is_valid_position(current_index):
                reward=-1e16
        elif action==self.action_goback:
            #Reward is distance to depot
            reward=self.current_customer_list[0][self.index_dist_to_depot]
            #Shift the list accordingly
            self.current_customer_list=self.current_customer_list[self.depot_index:]+self.current_customer_list[0:self.depot_index]
            #Refill truck capacity
            self.depot_index=self.current_customer_list.index(self.depot_location)
            self.current_truck_capacity=self.total_truck_capacity
        if len(self.unvisited)==0:
            isDone = True
        else:
            isDone = False
        #For now truncated=False until conditions are defined
        truncated = False
#        for i in range(len(self.visited) - 1):
#            if self.visited[i] == self.visited[i + 1]:
#                truncated = True
#        truncated=True
        # print(f'Here: {self.truck_current_location}, {reward}, {isDone}, {truncated}')    
        # return self.truck_current_location, reward, isDone, truncated, {"abc":1}
        state = []
        #Create flat state which are the inputs the neural network takes in
        #The neural network takes in the information for every customer
        #In total, 9 values for each customer
        for sublist in self.current_customer_list:
            state.extend([sublist[self.index_coord][0],sublist[self.index_coord][1],sublist[self.index_demand],sublist[self.index_dist_to_depot],sublist[self.index_dist_to_truck],sublist[self.index_pos_rel_depot][0],sublist[self.index_pos_rel_depot][1],sublist[self.index_pos_rel_truck][0],sublist[self.index_pos_rel_truck][1]])
        f_state = self.flat_list(state)
        return f_state, reward, isDone, truncated, {"visited":self.visited}

    def flat_list(self, mixed_list):
        f_list = [item for sublist in mixed_list for item in (sublist if isinstance(sublist, list) else [sublist])]
        return f_list
        
    def render(self):
        #printing out matrices
        state = []
        for sublist in self.current_customer_list:
            state.extend([sublist[0][0],sublist[0][1],sublist[1],sublist[2],sublist[3],sublist[4][0],sublist[4][1],sublist[5][0],sublist[5][1]])
        f_state = self.flat_list(state)
        print(f"States: {state}")
        print(f"States List: {f_state}")
        print(f"Visited: {self.visited}")
        print(f"Unvisited: {self.unvisited}")
        print(f"Current Customer List: {self.current_customer_list}")
        print(f"Truck Current Capacity: {self.current_truck_capacity}")
        print(f"Truck Max Capacity: {self.total_truck_capacity}")
        print(f"Depot Index: {self.depot_index}")
        print(f"Depot Location: {self.depot_location}")
