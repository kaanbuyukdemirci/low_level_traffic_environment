import torch
import cv2
import numpy as np
import copy
from typing import Callable, Literal
import gymnasium as gym

from f2_traffic_tensor import TrafficTensor
from commons import *

class LiveTrafficTensor(TrafficTensor):
    """A subclass of TrafficTensor that is used for live traffic simulation.
    More specifically, it is used to simulate a traffic environment for given actions.
    It does this through the act() method, which uses the brains given the state fro observation 
    to calculate the actions for each vehicle.
    
    Aside from init parameters, it has the following attributes (m is the number of vehicles):
        observation : torch.Tensor, shape=(m, 35)
            the observation of the current state. see update_observation() for more details.
        action : torch.Tensor, shape=(m, 2)
            the action of the current state
    """
    def __init__(self, x:torch.Tensor, y:torch.Tensor, width:torch.Tensor, length:torch.Tensor, angle:torch.Tensor, 
                 x_coverage:torch.Tensor, y_coverage:torch.Tensor, vehicle_line_thickness:float, 
                 vehicle_body_color:torch.Tensor, vehicle_head_color:torch.Tensor, 
                 lanes:torch.Tensor, lane_width:float, lane_line_thickness:float, lane_line_color:tuple[int,int,int], 
                 lane_boundary_line_thickness:float, lane_boundary_line_color:tuple[int,int,int], 
                 brains:list[Callable], brain_indexes:torch.Tensor, target_lanes:torch.Tensor, speed:torch.Tensor, 
                 time_interval:float, max_observable_distance:torch.Tensor, max_vehicle_speed:torch.Tensor, 
                 min_vehicle_speed:torch.Tensor, max_vehicle_acceleration:torch.Tensor,
                 min_vehicle_acceleration:torch.Tensor, max_vehicle_steering_angle:torch.Tensor,
                 dtype:torch.dtype, device:torch.device):
        """Constructor of LiveTrafficTensor.

        Parameters (m is the number of vehicles)
        ----------
        x : torch.Tensor, shape=(m,)
            The x coordinates of the vehicles.
        y : torch.Tensor, shape=(m,)
            The y coordinates of the vehicles.
        width : torch.Tensor, shape=(m,)
            The width of the vehicles.
        length : torch.Tensor, shape=(m,)
            The length of the vehicles.
        angle : torch.Tensor, shape=(m,)
            The angle of the vehicles.
        x_coverage : torch.Tensor, shape=(2,)
            The x coverage of the image.
        y_coverage : torch.Tensor, shape=(2,)
            The y coverage of the image.
        vehicle_line_thickness : float
            The thickness of the vehicle lines.
        vehicle_body_color : torch.Tensor, shape=(m,3)
            The color of the vehicle body.
        vehicle_head_color : torch.Tensor, shape=(m,3)
            The color of the vehicle head.
        lanes : torch.Tensor, shape=(n,4)
            The lane endpoints. Each lane is represented by a pair of points. 
            The first point is the start point and the second point is the end point.
            lanes[i, 0] is the x coordinate of the start point of lane i.
            lanes[i, 1] is the y coordinate of the start point of lane i.
            lanes[i, 2] is the x coordinate of the end point of lane i.
            lanes[i, 3] is the y coordinate of the end point of lane i.
            It is assumed that (i)th lane is to the left of (i+1)th lane.
            It is also assumed that the direction of the lane is from start to end point, meaning the
            vehicles should go that way. Otherwise, it might have unexpected effects on the observation.
        lane_width : float
            The width of the lanes. All lanes have the same width. 
            If a vehicle's distance to the closest lane is more than lane_width/2, 
            then it is considered to be out of lane.
        lane_line_thickness : float
            The thickness of the lane lines for the image.
        lane_line_color : tuple[int,int,int]
            The color of the lane lines for the image.
        lane_boundary_line_thickness : float
            The thickness of the lane boundary lines for the image.
        lane_boundary_line_color : tuple[int,int,int]
            The color of the lane boundary lines for the image.
        brains : list[Callable]
            List of functions that take in the observation and return the action.
        brain_indexes : torch.Tensor
            The indexes of the brains for each vehicle. A vehicle with index i will use the brain brains[i].
        target_lanes : torch.Tensor
            The target lanes of the vehicles. Each vehicle will try to reach its target lane.
        speed : torch.Tensor
            The speed of the vehicles.
        time_interval : float
            The time interval between each update.
        max_observable_distance : torch.tensor, shape=(m,)
            The maximum distance that a vehicle can observe.
        max_vehicle_speed : torch.Tensor
            The maximum speed of the vehicles.
        min_vehicle_speed : torch.Tensor
            The minimum speed of the vehicles.
        max_vehicle_acceleration : torch.Tensor
            The maximum acceleration of the vehicles.
        min_vehicle_acceleration : torch.Tensor
            The minimum acceleration of the vehicles.
        max_vehicle_steering_angle : torch.Tensor
            The maximum steering angle of the vehicles. The minimum steering angle is -max_vehicle_steering_angle.
        dtype : torch.dtype
            The data type of the tensors.
        device : torch.device
            The device of the tensors.
        """
        super().__init__(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color, lanes, lane_width, lane_line_thickness, lane_line_color, lane_boundary_line_thickness, lane_boundary_line_color, dtype, device)
        self.brains = brains
        self.brain_indexes = brain_indexes
        self.target_lanes = target_lanes
        self.speed = speed.to(dtype=dtype, device=device)
        self.time_interval = time_interval
        self.max_observable_distance = max_observable_distance.to(dtype=dtype, device=device)
        self.max_vehicle_speed = max_vehicle_speed.to(dtype=dtype, device=device)
        self.min_vehicle_speed = min_vehicle_speed.to(dtype=dtype, device=device)
        self.max_vehicle_acceleration = max_vehicle_acceleration.to(dtype=dtype, device=device)
        self.min_vehicle_acceleration = min_vehicle_acceleration.to(dtype=dtype, device=device)
        self.max_vehicle_steering_angle = max_vehicle_steering_angle.to(dtype=dtype, device=device)
        
        self.observation = None
        self.action = None
        
        self.__set_spaces()
        self.update()
        
    def __set_spaces(self):
        """Sets the observation and action spaces for each vehicle.
        """
        dtype = np_dtype(self.dtype)
        self.observation_spaces = []
        self.action_spaces = []
        number_of_vehicles = self.x.shape[0]
        
        # lane distances
        lane_back_distance_low = -self.max_observable_distance.numpy()
        lane_back_distance_high = np.zeros(number_of_vehicles, dtype=dtype)
        lane_front_distance_low = np.zeros(number_of_vehicles, dtype=dtype)
        lane_front_distance_high = self.max_observable_distance.numpy()
        
        # closest distances
        closest_distance_low = np.zeros(number_of_vehicles, dtype=dtype) - self.lane_width/2
        closest_distance_high = np.zeros(number_of_vehicles, dtype=dtype) + self.lane_width/2
        
        # angles
        angle_low = np.zeros(number_of_vehicles, dtype=dtype) - np.pi
        angle_high = np.zeros(number_of_vehicles, dtype=dtype) + np.pi
        
        # speed
        speed_low = self.min_vehicle_speed.numpy()
        speed_high = self.max_vehicle_speed.numpy()
        
        # relative speed TODO: it can get more precise
        relative_speed_low = speed_low - self.max_vehicle_speed.max().item()
        relative_speed_high = speed_high - self.min_vehicle_speed.min().item()
        
        # width
        width_low = self.width.numpy()
        width_high = self.width.numpy()
        
        # other width
        other_width_low = np.zeros(number_of_vehicles, dtype=dtype) + self.width.min().item()
        other_width_high = np.zeros(number_of_vehicles, dtype=dtype) + self.width.max().item()
        
        # length
        length_low = self.length.numpy()
        length_high = self.length.numpy()
        
        # other length
        other_length_low = np.zeros(number_of_vehicles, dtype=dtype) + self.length.min().item()
        other_length_high = np.zeros(number_of_vehicles, dtype=dtype) + self.length.max().item()
        
        # relative target lane
        relative_target_lane_low = np.zeros(number_of_vehicles, dtype=dtype) - (self.lanes.shape[0] - 1)
        relative_target_lane_high = np.zeros(number_of_vehicles, dtype=dtype) + self.lanes.shape[0] - 1
        
        # acceleration
        acceleration_low = self.min_vehicle_acceleration.numpy()
        acceleration_high = self.max_vehicle_acceleration.numpy()
        
        # steering angle
        steering_angle_low = -self.max_vehicle_steering_angle.numpy()
        steering_angle_high = self.max_vehicle_steering_angle.numpy()
        
        # all list
        all_lows = [relative_target_lane_low, speed_low, angle_low, width_low, length_low, closest_distance_low,
                    lane_back_distance_low, lane_front_distance_low, lane_back_distance_low, 
                    lane_front_distance_low, lane_back_distance_low, lane_front_distance_low,
                    relative_speed_low, relative_speed_low, relative_speed_low, 
                    relative_speed_low, relative_speed_low, relative_speed_low,
                    angle_low, angle_low, angle_low, angle_low, angle_low, angle_low,
                    other_width_low, other_width_low, other_width_low, 
                    other_width_low, other_width_low, other_width_low,
                    other_length_low, other_length_low, other_length_low,
                    other_length_low, other_length_low, other_length_low,
                    closest_distance_low, closest_distance_low, closest_distance_low,
                    closest_distance_low, closest_distance_low, closest_distance_low]
        all_highs = [relative_target_lane_high, speed_high, angle_high, width_high, length_high, closest_distance_high,
                     lane_back_distance_high, lane_front_distance_high, lane_back_distance_high,
                     lane_front_distance_high, lane_back_distance_high, lane_front_distance_high,
                     relative_speed_high, relative_speed_high, relative_speed_high,
                     relative_speed_high, relative_speed_high, relative_speed_high,
                     angle_high, angle_high, angle_high, angle_high, angle_high, angle_high,
                     other_width_high, other_width_high, other_width_high,
                     other_width_high, other_width_high, other_width_high,
                     other_length_high, other_length_high, other_length_high,
                     other_length_high, other_length_high, other_length_high,
                     closest_distance_high, closest_distance_high, closest_distance_high,
                     closest_distance_high, closest_distance_high, closest_distance_high]
        
        # calculate all spaces
        observation_spaces = []
        action_spaces = []
        for i in range(number_of_vehicles):
            low_array = np.array([x[i] for x in all_lows], dtype=dtype)
            high_array = np.array([x[i] for x in all_highs], dtype=dtype)
            # observation space
            observation_space = gym.spaces.Box(low=low_array, high=high_array, 
                                               shape=(42,), 
                                               dtype=dtype)
                
            # action space
            action_space = gym.spaces.Box(
                low=np.array([acceleration_low[i], steering_angle_low[i]]), 
                high=np.array([acceleration_high[i], steering_angle_high[i]]), 
                dtype=dtype)
            
            observation_spaces.append(observation_space)
            action_spaces.append(action_space)
        
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
    
    def update(self):
        """Updates the state of the vehicles. This method should be called
        every time one of the init parameters is updated.
        """
        super().update()
        self.update_observation()
    
    def __observe_lane(self, lane_index:Literal['left','same','right']) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Observe the given lane and return the distances to the closest vehicle in front&back.

        Parameters
        ----------
        lane_index : Literal['left','same','right']
            The lane to observe. 
            'left' means the lane to the left of the vehicle, 
            'same' means the same lane as the vehicle, 
            'right' means the lane to the right of the vehicle.

        Returns (m is the number of vehicles)
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], shape=(m,) for each tensor
            back_index : torch.Tensor
                The indexes of the closest vehicle in the back.
            back_distance : torch.Tensor
                The distances to the closest vehicle in the back.
            back_mask : torch.Tensor
                A mask indicating whether there is a vehicle in the back.
                If there is no vehicle in the back, then the distance is set to -max_observable_distance,
                and the mask is set to True.
            front_index : torch.Tensor
                The indexes of the closest vehicle in the front.
            front_distance : torch.Tensor
                The distances to the closest vehicle in the front.
            front_mask : torch.Tensor
                A mask indicating whether there is a vehicle in the front.
                If there is no vehicle in the front, then the distance is set to max_observable_distance,
                and the mask is set to True.
        """
        # adjust lane indexes
        if lane_index == 'left':
            lane_index = self.closest_lanes.view(-1, 1) - 1
        elif lane_index == 'same':
            lane_index = self.closest_lanes.view(-1, 1)
        elif lane_index == 'right':
            lane_index = self.closest_lanes.view(-1, 1) + 1
        
        # adjust lane_positions and lane_index so that we don't have negative lane indexes or indexes that are too big
        extended_lane_index = lane_index + 1
        extended_lane_positions = torch.cat((torch.zeros(size=(lane_index.shape[0], 1)), 
                                             self.lane_positions, 
                                             torch.zeros(size=(lane_index.shape[0], 1))), 
                                            dim=1)
        
        # find the mask for the vehicles in the lane and ignore the vehicle itself (diagonal elements), also find the distances
        in_lane_vehicle_mask = lane_index == self.closest_lanes.view(1, -1)
        in_lane_vehicle_mask[torch.arange(in_lane_vehicle_mask.shape[0]), torch.arange(in_lane_vehicle_mask.shape[0])] = False
        left = self.lane_position.view(1, -1) # you can calculate this like right, it doesn't matter thanks to the mask
        right = extended_lane_positions[torch.arange(extended_lane_positions.shape[0]), extended_lane_index.view(-1)].view(-1, 1)
        
        # apply mask to get the true distances
        masked_distances = left - right
        masked_distances[~in_lane_vehicle_mask] = torch.inf
        masked_distances = torch.nan_to_num(masked_distances, nan=torch.inf)
        
        # get the closest back and front distance
        inverse_distances = 1/masked_distances
        back_index = torch.argmin(inverse_distances, dim=1)
        back_distance = -torch.abs(masked_distances[torch.arange(masked_distances.shape[0]), back_index])
        back_distance = torch.clamp(back_distance, min=-self.max_observable_distance)
        front_index = torch.argmax(inverse_distances, dim=1)
        front_distance = masked_distances[torch.arange(masked_distances.shape[0]), front_index]
        front_distance = torch.clamp(front_distance, max=self.max_observable_distance)
        
        # masks indicating whether there is a vehicle in front&back
        back_mask = back_distance == -self.max_observable_distance
        front_mask = front_distance == self.max_observable_distance
        
        # other stuff
        return back_index, back_distance, back_mask, front_index, front_distance, front_mask
    
    def update_observation(self):
        """Updates the observation for each vehicle. The observation is:
            self_relative_target_lane : torch.Tensor, shape=(m,)
                The relative target lane of each vehicle.
            self_speed : torch.Tensor, shape=(m,)
                The speed of each vehicle.
            self_angle : torch.Tensor, shape=(m,)
                The angle of each vehicle.
            self_width : torch.Tensor, shape=(m,)
                The width of each vehicle.
            self_length : torch.Tensor, shape=(m,)
                The length of each vehicle.
            self_closest_distance : torch.Tensor, shape=(m,)
                The closest distance to the closest vehicle.
            relative_distance : torch.Tensor, shape=(m,6)
                The relative distance to the closest vehicle in back&front in left&same&right lane.
            relative_speed : torch.Tensor, shape=(m,6)
                The relative speed to the closest vehicle in back&front in left&same&right lane.
            relative_angle: torch.Tensor, shape=(m,6)
                The relative angle to the closest vehicle in back&front in left&same&right lane.
            relative_width : torch.Tensor, shape=(m,6)
                The relative width to the closest vehicle in back&front in left&same&right lane.
            relative_length : torch.Tensor, shape=(m,6)
                The relative length to the closest vehicle in back&front in left&same&right lane.
            closest_distance : torch.Tensor, shape=(m,6)
                The closest distance to the closest vehicle in back&front in left&same&right lane.
            
            Final observation is:
            observation : torch.Tensor, shape=(m,42)
                The concatenation of all the above tensors.
        """
        # relative target lane
        relative_target_lanes = self.target_lanes - self.closest_lanes
        # speed
        speed = self.speed
        # angle
        angle = self.angle % 2*torch.pi
        # width
        width = self.width
        # length
        length = self.length
        # distance
        closest_distance = self.closest_distances
        
        # left lane ------------------------------------------------------------------------------------------
        back_index, back_distance, back_mask, front_index, front_distance, front_mask = self.__observe_lane('left')
        
        left_lane_back_distance = back_distance
        left_lane_back_speed = self.speed[back_index] - self.speed
        left_lane_back_speed[back_mask] = 0
        left_lane_back_angle = (self.angle[back_index] - self.angle)% 2*torch.pi
        left_lane_back_angle[left_lane_back_angle > torch.pi] -= 2*torch.pi
        left_lane_back_angle[back_mask] = 0
        left_lane_back_width = self.width[back_index]
        left_lane_back_width[back_mask] = self.width[back_mask]
        left_lane_back_length = self.length[back_index]
        left_lane_back_length[back_mask] = self.length[back_mask]
        left_lane_back_closest_distance = self.closest_distances[back_index]
        left_lane_back_closest_distance[back_mask] = 0
        
        left_lane_front_distance = front_distance
        left_lane_front_speed = self.speed[front_index] - self.speed
        left_lane_front_speed[front_mask] = 0
        left_lane_front_angle = (self.angle[front_index] - self.angle) % 2*torch.pi
        left_lane_front_angle[left_lane_front_angle > torch.pi] -= 2*torch.pi
        left_lane_front_angle[front_mask] = 0
        left_lane_front_width = self.width[front_index]
        left_lane_front_width[front_mask] = self.width[front_mask]
        left_lane_front_length = self.length[front_index]
        left_lane_front_length[front_mask] = self.length[front_mask]
        left_lane_front_closest_distance = self.closest_distances[front_index]
        left_lane_front_closest_distance[front_mask] = 0
        
        # same lane ------------------------------------------------------------------------------------------
        back_index, back_distance, back_mask, front_index, front_distance, front_mask = self.__observe_lane('same')
        
        same_lane_back_distance = back_distance
        same_lane_back_speed = self.speed[back_index] - self.speed
        same_lane_back_speed[back_mask] = 0
        same_lane_back_angle = (self.angle[back_index] - self.angle) % 2*torch.pi
        same_lane_back_angle[same_lane_back_angle > torch.pi] -= 2*torch.pi
        same_lane_back_angle[back_mask] = 0
        same_lane_back_width = self.width[back_index]
        same_lane_back_width[back_mask] = self.width[back_mask]
        same_lane_back_length = self.length[back_index]
        same_lane_back_length[back_mask] = self.length[back_mask]
        same_lane_back_closest_distance = self.closest_distances[back_index]
        same_lane_back_closest_distance[back_mask] = 0
        
        same_lane_front_distance = front_distance
        same_lane_front_speed = self.speed[front_index] - self.speed
        same_lane_front_speed[front_mask] = 0
        same_lane_front_angle = (self.angle[front_index] - self.angle) % 2*torch.pi
        same_lane_front_angle[same_lane_front_angle > torch.pi] -= 2*torch.pi
        same_lane_front_angle[front_mask] = 0
        same_lane_front_width = self.width[front_index]
        same_lane_front_width[front_mask] = self.width[front_mask]
        same_lane_front_length = self.length[front_index]
        same_lane_front_length[front_mask] = self.length[front_mask]
        same_lane_front_closest_distance = self.closest_distances[front_index]
        same_lane_front_closest_distance[front_mask] = 0
        
        # right lane ------------------------------------------------------------------------------------------
        back_index, back_distance, back_mask, front_index, front_distance, front_mask = self.__observe_lane('right')
        
        right_lane_back_distance = back_distance
        right_lane_back_speed = self.speed[back_index] - self.speed
        right_lane_back_speed[back_mask] = 0
        right_lane_back_angle = (self.angle[back_index] - self.angle) % 2*torch.pi
        right_lane_back_angle[right_lane_back_angle > torch.pi] -= 2*torch.pi
        right_lane_back_angle[back_mask] = 0
        right_lane_back_width = self.width[back_index]
        right_lane_back_width[back_mask] = self.width[back_mask]
        right_lane_back_length = self.length[back_index]
        right_lane_back_length[back_mask] = self.length[back_mask]
        right_lane_back_closest_distance = self.closest_distances[back_index]
        right_lane_back_closest_distance[back_mask] = 0
        
        right_lane_front_distance = front_distance
        right_lane_front_speed = self.speed[front_index] - self.speed
        right_lane_front_speed[front_mask] = 0
        right_lane_front_angle = (self.angle[front_index] - self.angle) % 2*torch.pi
        right_lane_front_angle[right_lane_front_angle > torch.pi] -= 2*torch.pi
        right_lane_front_angle[front_mask] = 0
        right_lane_front_width = self.width[front_index]
        right_lane_front_width[front_mask] = self.width[front_mask]
        right_lane_front_length = self.length[front_index]
        right_lane_front_length[front_mask] = self.length[front_mask]
        right_lane_front_closest_distance = self.closest_distances[front_index]
        right_lane_front_closest_distance[front_mask] = 0
        
        # concat everything
        self_values = torch.stack((relative_target_lanes, speed, angle, width, length, closest_distance), dim=1)
        distance = torch.stack((left_lane_back_distance, left_lane_front_distance,
                                same_lane_back_distance, same_lane_front_distance, 
                                right_lane_back_distance, right_lane_front_distance), dim=1)
        speed = torch.stack((left_lane_back_speed, left_lane_front_speed,
                             same_lane_back_speed, same_lane_front_speed,
                             right_lane_back_speed, right_lane_front_speed), dim=1)
        angle = torch.stack((left_lane_back_angle, left_lane_front_angle,
                             same_lane_back_angle, same_lane_front_angle,
                             right_lane_back_angle, right_lane_front_angle), dim=1)
        width = torch.stack((left_lane_back_width, left_lane_front_width,
                             same_lane_back_width, same_lane_front_width,
                             right_lane_back_width, right_lane_front_width), dim=1)
        length = torch.stack((left_lane_back_length, left_lane_front_length,
                              same_lane_back_length, same_lane_front_length,
                              right_lane_back_length, right_lane_front_length), dim=1)
        closest_distance = torch.stack((left_lane_back_closest_distance, left_lane_front_closest_distance,
                                         same_lane_back_closest_distance, same_lane_front_closest_distance,
                                         right_lane_back_closest_distance, right_lane_front_closest_distance), dim=1)
        
        return_tensor = torch.cat((self_values, distance, speed, angle, width, length, closest_distance), dim=1)
        # shape is (vehicle count, 35)
        
        return_tensor = return_tensor.numpy()
        self.observation = return_tensor
        return return_tensor

    def act(self, agent_action:torch.Tensor, agent_brain_index:int=-1):
        """Execute the given actions for the agents, and execute the brains for the other vehicles.
        
        Parameters
        ----------
        agent_action : torch.Tensor, shape (agent count, 2)
            The actions for the agents. The number of agents is equal to the number of brain indexes that are equal to agent_brain_index.
            agent_action[:, 0] is the acceleration.
            agent_action[:, 1] is the steering angle.
        agent_brain_index : int, optional
            The brain index of the agents. The default is -1.
        """
        # define action: 2D tensor with shape (vehicle count, 2)
        # 2 is for acceleration and steering angle
        action = torch.zeros(size=(self.speed.shape[0], 2))
        
        # execute each brain only once
        for brain_index, brain in enumerate(self.brains):
            indexes = torch.nonzero(self.brain_indexes == brain_index, as_tuple=True)[0]
            other_indexes = torch.arange(indexes.shape[0])
            states = self.observation[indexes]
            actions = brain(states)
            action[indexes] = actions[other_indexes]
        
        # take the agent action into account
        indexes = torch.nonzero(self.brain_indexes == agent_brain_index, as_tuple=True)[0]
        #other_indexes = torch.arange(indexes.shape[0])
        action[indexes] = agent_action#.view(-1, action.shape[1])[other_indexes]
        
        # save action
        self.action = action
        
        # take the action
        vehicle_acceleration = action[:,0]
        vehicle_steering_angle = action[:,1]
        
        # beta
        beta = torch.complex(2*torch.ones_like(vehicle_steering_angle), torch.tan(vehicle_steering_angle))
        beta = torch.angle(beta)
        
        # speed
        vehicle_acceleration = torch.minimum(vehicle_acceleration, self.max_vehicle_acceleration)
        vehicle_acceleration = torch.maximum(vehicle_acceleration, self.min_vehicle_acceleration)
        speed = torch.minimum(self.speed + self.time_interval*vehicle_acceleration, self.max_vehicle_speed)
        speed = torch.maximum(speed, self.min_vehicle_speed)
        self.speed = speed
        
        # angle
        vehicle_steering_angle = torch.minimum(vehicle_steering_angle, self.max_vehicle_steering_angle)
        vehicle_steering_angle = torch.maximum(vehicle_steering_angle, -self.max_vehicle_steering_angle)
        self.angle += self.time_interval * (self.speed / self.length) * torch.cos(beta) * torch.tan(vehicle_steering_angle)
        self.angle %= (2 * torch.pi)
        
        # position
        self.x += self.time_interval * self.speed * torch.cos(self.angle + beta)
        self.y += self.time_interval * self.speed * torch.sin(self.angle + beta)
        
        # update the rectangle
        self.update()

    def copy(self):
        return copy.deepcopy(self)

def test1():
    """To test the LiveTrafficTensor class in general.
    """
    x = torch.tensor([0.0, 0.0, 0.0])
    y = torch.tensor([10.0, 0.0, -9.0])
    width = torch.tensor([5.0, 6.0, 7.0])
    length = torch.tensor([10.0, 12.0, 14.0])
    angle = torch.tensor([0.0, 0.0, torch.pi/4])
    x_coverage = torch.tensor([-100.0, 100.0])
    y_coverage = torch.tensor([-20.0, 20.0])
    vehicle_line_thickness = 1
    vehicle_body_color = torch.tensor([255, 0, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    vehicle_head_color = torch.tensor([255, 255, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    dtype = torch.float32
    device = torch.device("cpu")
    
    lanes = torch.tensor([[0.0, 10.0, 10.0, 10.0],
                            [0.0, -10.0, 10.0, -10.0]])
    lane_width = 10.0
    lane_line_thickness = 1
    lane_line_color = (0, 255, 0)
    lane_boundary_line_thickness = 1
    lane_boundary_line_color = (255, 0, 255)
    
    brains = [lambda obs: torch.zeros(size=(obs.shape[0], 2))]
    brain_indexes = torch.tensor([-1, -1, 0])
    target_lanes = torch.tensor([0, 1, 1])
    speed = torch.tensor([1, 2, 3])
    time_interval = 1.0
    max_observed_distance = torch.zeros(x.shape[0]) + 100.0
    max_vehicle_speed = torch.tensor([100, 100, 100])
    min_vehicle_speed = torch.tensor([-1, 0, 0])
    max_vehicle_acceleration = torch.tensor([5, 5, 5])
    min_vehicle_acceleration = torch.tensor([-5, -5, -5])
    max_vehicle_steering_angle = torch.tensor([torch.pi/4, torch.pi/4, torch.pi/4])
    
    live_traffic_tensor = LiveTrafficTensor(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color,
                                            lanes, lane_width, lane_line_thickness, lane_line_color, lane_boundary_line_thickness, lane_boundary_line_color,
                                            brains, brain_indexes, target_lanes, speed, time_interval, max_observed_distance, max_vehicle_speed, min_vehicle_speed, max_vehicle_acceleration, min_vehicle_acceleration,
                                            max_vehicle_steering_angle, dtype, device)
    
    #live_traffic_tensor.observe()
    #canvas = np.zeros((300, 1500, 3), dtype=np.uint8)
    #canvas = live_traffic_tensor.render(canvas)
    #cv2.imshow("canvas", canvas)
    #cv2.waitKey(0)
    
    speed_up_rate = 100
    for i in range(100):
        # canvas
        canvas = np.zeros((300, 1500, 3), dtype=np.uint8)
        
        # step
        action = torch.zeros(size=(2,2))
        action[:,1] = -torch.pi/10
        live_traffic_tensor.act(action)
        
        # render
        canvas = live_traffic_tensor.render(canvas)
        cv2.imshow('test', canvas)
        cv2.waitKey(int(live_traffic_tensor.time_interval*speed_up_rate))

def test2():
    """To test the observations.
    """
    x = torch.tensor([0.0, 0.0, 0.0])
    y = torch.tensor([10.0, 0.0, -9.0])
    width = torch.tensor([5.0, 6.0, 7.0])
    length = torch.tensor([10.0, 12.0, 14.0])
    angle = torch.tensor([0.0, 0.0, torch.pi/4])
    x_coverage = torch.tensor([-100.0, 100.0])
    y_coverage = torch.tensor([-20.0, 20.0])
    vehicle_line_thickness = 1
    vehicle_body_color = torch.tensor([255, 0, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    vehicle_head_color = torch.tensor([255, 255, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    dtype = torch.float32
    device = torch.device("cpu")
    
    lanes = torch.tensor([[0.0, 10.0, 10.0, 10.0],
                            [0.0, -10.0, 10.0, -10.0]])
    lane_width = 10.0
    lane_line_thickness = 1
    lane_line_color = (0, 255, 0)
    lane_boundary_line_thickness = 1
    lane_boundary_line_color = (255, 0, 255)
    
    brains = [lambda obs: torch.zeros(size=(obs.shape[0], 2))]
    brain_indexes = torch.tensor([-1, 0, 0])
    target_lanes = torch.tensor([0, 1, 1])
    speed = torch.tensor([1, 2, 3])
    time_interval = 1.0
    max_observed_distance = torch.zeros(x.shape[0]) + 100.0
    max_vehicle_speed = torch.tensor([100, 100, 100])
    min_vehicle_speed = torch.tensor([-1, 0, 0])
    max_vehicle_acceleration = torch.tensor([5, 5, 5])
    min_vehicle_acceleration = torch.tensor([-5, -5, -5])
    max_vehicle_steering_angle = torch.tensor([torch.pi/4, torch.pi/4, torch.pi/4])
    
    live_traffic_tensor = LiveTrafficTensor(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, 
                                            vehicle_head_color, lanes, lane_width, lane_line_thickness, lane_line_color, lane_boundary_line_thickness, 
                                            lane_boundary_line_color, brains, brain_indexes, target_lanes, speed, time_interval, max_observed_distance, 
                                            max_vehicle_speed, min_vehicle_speed, max_vehicle_acceleration, min_vehicle_acceleration, max_vehicle_steering_angle, 
                                            dtype, device)
    
    for i in range(100):
        # canvas
        canvas = np.zeros((300, 1500, 3), dtype=np.uint8)
        
        # step
        action = torch.zeros(size=(2,))
        action[1] = -torch.pi/10
        live_traffic_tensor.act(action)
        observation_keys = ["self_relative_target_lane", "self_speed", "self_angle", "self_width", "self_length", "self_closest_distance",
                            "relative_distance_1", "relative_distance_2", "relative_distance_3", "relative_distance_4", "relative_distance_5", "relative_distance_6",
                            "relative_speed_1", "relative_speed_2", "relative_speed_3", "relative_speed_4", "relative_speed_5", "relative_speed_6", 
                            "relative_angle_1", "relative_angle_2", "relative_angle_3", "relative_angle_4", "relative_angle_5", "relative_angle_6", 
                            "relative_width_1", "relative_width_2", "relative_width_3", "relative_width_4", "relative_width_5", "relative_width_6", 
                            "relative_length_1", "relative_length_2", "relative_length_3", "relative_length_4", "relative_length_5", "relative_length_6",
                            "closest_distance_1", "closest_distance_2", "closest_distance_3", "closest_distance_4", "closest_distance_5", "closest_distance_6"]
        observation = dict()
        vehicle_index = 1
        for i, key in enumerate(observation_keys):
            observation[key] = live_traffic_tensor.observation[vehicle_index,i]
        new_line_or_tab = False
        for key in observation_keys:
            print(key, observation[key], end=("\n" if new_line_or_tab else "\t\t\t"))
            new_line_or_tab = not new_line_or_tab
        print(live_traffic_tensor.closest_lanes[vehicle_index])
        print()
        
        # render
        canvas = live_traffic_tensor.render(canvas)
        cv2.imshow('test', canvas)
        cv2.waitKey()

if __name__ == "__main__":
    #test1()
    test2()
    pass