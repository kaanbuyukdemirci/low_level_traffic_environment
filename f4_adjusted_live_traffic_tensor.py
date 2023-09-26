import torch
from typing import Callable
import cv2
import numpy as np

from f3_live_traffic_tensor import LiveTrafficTensor

class AdjustedLiveTrafficTensor(LiveTrafficTensor): 
    """A class that represents a live traffic tensor with adjusted observations and actions.
    It adjusts the observations and actions in according to user's needs.
    
    Aside from the this adjustment, this class is the same as LiveTrafficTensor.
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
                 dtype:torch.dtype, device:torch.device, observation_adjuster:Callable=None, 
                 observation_space_fixer:Callable=None, action_adjuster:Callable=None,
                 action_space_fixer:Callable=None):
        """Constructor of DiscreteLiveTrafficTensor.

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
        brain_indexes : torch.Tensor, shape=(m,)
            The indexes of the brains for each vehicle. A vehicle with index i will use the brain brains[i].
        target_lanes : torch.Tensor, shape=(m,)
            The target lanes of the vehicles. Each vehicle will try to reach its target lane.
        speed : torch.Tensor, shape=(m,)
            The speed of the vehicles.
        time_interval : float
            The time interval between each update.
        max_observable_distance : torch.Tensor, shape=(m,)
            The maximum distance that a vehicle can observe.
        max_vehicle_speed : torch.Tensor, shape=(m,)
            The maximum speed of the vehicles.
        min_vehicle_speed : torch.Tensor, shape=(m,)
            The minimum speed of the vehicles.
        max_vehicle_acceleration : torch.Tensor, shape=(m,)
            The maximum acceleration of the vehicles.
        min_vehicle_acceleration : torch.Tensor, shape=(m,)
            The minimum acceleration of the vehicles.
        max_vehicle_steering_angle : torch.Tensor, shape=(m,)
            The maximum steering angle of the vehicles. The minimum steering angle is -max_vehicle_steering_angle.
        dtype : torch.dtype
            The data type of the tensors.
        device : torch.device
            The device of the tensors.
        observation_adjuster : Callable, optional
            The function that adjusts the observation. The default is None.
            If None, then no adjustment will be made.
        observation_space_fixer : Callable, optional
            The function that fixes the observation space. The default is None.
        action_adjuster : Callable, optional
            The function that adjusts the action. The default is None.
            If None, then no adjustment will be made.
        action_space_fixer : Callable, optional
            The function that fixes the action space. The default is None.
        """
        self.observation_adjuster = observation_adjuster
        self.observation_space_fixer = observation_space_fixer
        self.action_adjuster = action_adjuster
        self.action_space_fixer = action_space_fixer
        super().__init__(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color,
                         lanes, lane_width, lane_line_thickness, lane_line_color, lane_boundary_line_thickness, lane_boundary_line_color,
                         brains, brain_indexes, target_lanes, speed, time_interval, max_observable_distance, max_vehicle_speed, min_vehicle_speed, 
                         max_vehicle_acceleration, min_vehicle_acceleration, max_vehicle_steering_angle, dtype, device)
        if self.observation_space_fixer is not None:
            self.observation_spaces = self.observation_space_fixer(self.observation_spaces)
        if self.action_space_fixer is not None:
            self.action_spaces = self.action_space_fixer(self.action_spaces)
        
    def update(self):
        """Updates the live traffic tensor.
        """
        super().update()
        self.update_observation()
    
    def update_observation(self):
        """Updates the observation.
        """
        super().update_observation()
        if self.observation_adjuster is not None:
            self.observation = self.observation_adjuster(self.observation)
        return self.observation

    def act(self, agent_action:torch.Tensor, agent_brain_index:int=-1):
        """Acts on the live traffic tensor.

        Parameters
        ----------
        agent_action : torch.Tensor
            The action of the agent.
        agent_brain_index : int, optional
            The index of the brain of the agent. The default is -1.
        """
        if self.action_adjuster is not None:
            agent_action = self.action_adjuster(agent_action)
        super().act(agent_action, agent_brain_index)

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
    
    live_traffic_tensor = AdjustedLiveTrafficTensor(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color,
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
    
    live_traffic_tensor = AdjustedLiveTrafficTensor(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, 
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
    test1()
    test2()
    pass