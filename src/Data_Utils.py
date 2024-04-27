import random as r
import math as m

def normalized_instance_creator(cust_max_dem, truck_cap, num_cust):
    depot = (r.random(),r.random())
    occupied_coords=[]
    occupied_coords.append(depot)
    current_customer_list=[]
    current_customer_list.append(depot)
    for cust in range(num_cust):
        customer=[]
        coord=depot
        while coord in occupied_coords:
            coord=(r.random(),r.random())
        occupied_coords.append(coord)
        customer.append(coord)
        demand=(r.randint(1,cust_max_dem))
        customer.append(demand)
        distance_to_depot=dist = (m.sqrt(((depot[0]-coord[0])**2)+(depot[1]-coord[1])**2))
        customer.append(distance_to_depot)
        distance_to_truck=distance_to_depot
        customer.append(distance_to_truck)
        position_to_depot=(coord[0]-depot[0],coord[1]-depot[1])
        customer.append(position_to_depot)
        position_to_truck=position_to_depot
        customer.append(position_to_truck)
        current_customer_list.append(customer)
    current_truck_capacity=truck_cap
    total_truck_capacity=truck_cap
    return current_customer_list, current_truck_capacity, total_truck_capacity, depot

def create_data_set(num_cust, max_cust_capacity, truck_capacity, num_samples, seed=None):
    #Set seed
    r.seed(seed)
    #Init list to store samples
    samples = []
    for i in range(num_samples):
        current_customer_list, current_truck_capacity, total_truck_capacity, depot_location \
        = normalized_instance_creator(max_cust_capacity, truck_capacity, num_cust)
        samples.append({"current_customer_list":current_customer_list, "current_truck_capacity":current_truck_capacity, \
                        "total_truck_capacity":total_truck_capacity, "depot_location":depot_location})
    return samples

class dataGenerator:
    def __init__(self, num_cust, max_cust_capacity, truck_capacity, num_samples, data_path, data_prefix, seed=None):
        self.test_data_all = create_data_set(num_cust, max_cust_capacity, truck_capacity, num_samples, seed)
        self.num_cust = num_cust
        self.max_cust_capacity = max_cust_capacity
        self.truck_capacity = truck_capacity
        self.num_samples = num_samples
        self.seed = seed

    def get_num_samples(self):
        return self.num_samples
        
    def get_test_data_all(self):
        return self.test_data_all

    def get_test_data_ranged(self, start, num):
        return self.test_data_all[start:start+num]
    
    def get_train_next(self):
        self.train_data = create_data_set(self.num_cust, self.max_cust_capacity, self.truck_capacity, 1, self.seed)
        return self.train_data
    
    def get_test_next(self):
        #Start idx is random integer from 0 to the length of test data set minus number of samples and minus one because randint is inclusive
        start_idx = r.randint(0, len(self.test_data_all)-self.num_samples-1)
        self.test_data = self.test_data_all[start_idx:start_idx+self.num_samples]
        return start_idx, self.test_data


class dataPerformance:
    def __init__(self, data, num_samples):
        self.num_samples=num_samples
        self.data=data
        # keep the performance of each sample as best_episode, best_cost, best_path, max_cost
        self.performance=[[0,'undefined',[],'undefined'] for _ in range(len(data))]
        for idx in range(len(self.performance)):
            current_customer_list = self.data[idx]["current_customer_list"]
            depot_location = self.data[idx]["depot_location"]
            max_dist = 0
            for customer in current_customer_list:
                if customer!=depot_location:
                    max_dist+=2*customer[2]
            self.performance[idx][3] = max_dist
    
    def update_sample_performance(self, idx, best_episode, best_cost, best_path):
        self.performance[idx] = [best_episode, best_cost, best_path,self.performance[idx][3]]

    def get_sample_performance(self, idx):
        return self.performance[idx]
    





