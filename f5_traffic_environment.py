import torch
import cv2
import numpy as np
import gym
from typing import Callable, Any

from f4_adjusted_live_traffic_tensor import AdjustedLiveTrafficTensor

# https://www.programiz.com/python-programming/methods/built-in/super
class TrafficEnvironment(AdjustedLiveTrafficTensor, gym.Env):
    """The traffic environment class. It is a subclass of the gym.Env class and the AdjustedLiveTrafficTensor class.
    It is a digital environment that simulates traffic. It is a multi-agent environment. 
    TODO: it is a single agent environment rn, but it will be a multi-agent environment in the future.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, width: torch.Tensor, length: torch.Tensor, 
                 angle: torch.Tensor, x_coverage: torch.Tensor, y_coverage: torch.Tensor, 
                 vehicle_line_thickness: float, vehicle_body_color: torch.Tensor, 
                 vehicle_head_color: torch.Tensor, lanes: torch.Tensor, lane_width: float, 
                 lane_line_thickness: float, lane_line_color: tuple[int, int, int], 
                 lane_boundary_line_thickness: float, lane_boundary_line_color: tuple[int, int, int], 
                 brains: list[Callable[..., Any]], brain_indexes: torch.Tensor, target_lanes: torch.Tensor, 
                 speed: torch.Tensor, time_interval: float, max_observable_distance: torch.Tensor, 
                 max_vehicle_speed: torch.Tensor, min_vehicle_speed: torch.Tensor, 
                 max_vehicle_acceleration: torch.Tensor, min_vehicle_acceleration: torch.Tensor, 
                 max_vehicle_steering_angle: torch.Tensor, dtype: torch.dtype, device: torch.device, 
                 observation_digitalizer: Callable[..., Any] = None, 
                 observation_space_fixer: Callable[..., Any] = None, 
                 action_digitalizer: Callable[..., Any] = None, action_space_fixer: Callable[..., Any] = None, 
                 agent_index: int = 0, agent_brain_index:int=-1, max_step_count:int=None):
        """_summary_

        Parameters
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
        observation_digitalizer : Callable, optional
            The function that digitizes the observation. The default is None.
            If None, then no digitizer will be used.
        observation_space_fixer : Callable, optional
            The function that fixes the observation space. The default is None.
        action_digitalizer : Callable, optional
            The function that digitizes the action. The default is None.
            If None, then no digitizer will be used.
        action_space_fixer : Callable, optional
            The function that fixes the action space. The default is None.
        agent_index : int, optional
            The index of the agent in the list of agents. The default is 0.
        agent_brain_index : int, optional
            The index of the brain that the agent uses. The default is -1.
        max_step_count : int, optional
            The maximum number of steps that the agent can take. The default is None,
            which corresponds to default 10 seconds worth of steps.
        """
        super().__init__(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, 
                         vehicle_body_color, vehicle_head_color, lanes, lane_width, lane_line_thickness, 
                         lane_line_color, lane_boundary_line_thickness, lane_boundary_line_color, brains, 
                         brain_indexes, target_lanes, speed, time_interval, max_observable_distance, 
                         max_vehicle_speed, min_vehicle_speed, max_vehicle_acceleration, 
                         min_vehicle_acceleration, max_vehicle_steering_angle, dtype, device, 
                         observation_digitalizer, observation_space_fixer, action_digitalizer, 
                         action_space_fixer)
        # agent index is the index of the agent in the list of agents
        self.agent_index = agent_index
        self.agent_brain_index = agent_brain_index
        
        # make sure agent_index and agent_brain_index are compatible
        self.brain_indexes[self.agent_index] = self.agent_brain_index
        if (self.brain_indexes == -1).sum().item() > 1:
            #TODO: it will support multi agent in future
            raise ValueError("There can only be one agent brain index that is -1.")
        
        # it is assumed that only these values change during simulation, and other dependent values:
        self.initial_x = x
        self.initial_y = y
        self.initial_speed = speed
        self.initial_angle = angle
        
        # extra variables
        if max_step_count is None:
            max_step_count = int((1/self.time_interval)*10) # 10 seconds
        self.max_step_count = max_step_count
        self.step_counter = None
        
        # spaces
        self.observation_space = self.observation_spaces[self.agent_index]
        self.action_space = self.action_spaces[self.agent_index]
        
    def step(self, action: torch.Tensor):
        """The step function of the environment. It takes in an action and returns the observation, reward, done, and info.

        Parameters
        ----------
        action : torch.Tensor
            The action to take.


        Returns
        -------
        obs : torch.Tensor
            The observation.
        reward : float
            The reward.
        done : bool
            Whether the episode is done.
        info : dict
            The info.
        
        The action space is:
            acceleration : torch.Tensor, shape=(m,)
                The acceleration of each vehicle.
            steering_angle : torch.Tensor, shape=(m,)
                The steering angle of each vehicle.
            
        
        The observation space is:
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
        # take action
        self.act(action, self.agent_brain_index)
        
        # calculate obs
        obs = self.observation[self.agent_index]
        
        # calculate reward
        reward = 0
        
        # done and step counter
        self.step_counter += 1
        done = self.truncated() or self.terminated()
        
        return obs, reward, done, {}
        
    def terminated(self):
        if (self.intersection_vector[self.agent_index].sum() > 0) or \
        ((~self.is_in_lane)[self.agent_index].sum() > 0): # 'collision' or 'out of lane'
            return True
        else:
            return False
    
    def truncated(self):
        if self.step_counter >= self.max_step_count:
            return True
        else:
            return False

    def reset(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.speed = self.initial_speed
        self.angle = self.initial_angle
        self.step_counter = 0
        obs = self.observation[self.agent_index]
        return obs
    
    def render(self, image):
        image = super().render(image, mirror=True)
        return image
    
    def close(self):
        pass
    
def test1():
    x = torch.tensor([0.0, 0.0, 0.0, -20, -20, -30, -30, -40, -40])
    y = torch.tensor([10.0, 0.0, -4.0, 0.0, -4.0, 0.0, -4.0, 1.0, -4.0])
    width = torch.tensor([2.5, 2.5, 2.5, 2.5, 2.5, 4.0, 2.5, 2.5, 4.0])
    length = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0])
    angle = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_coverage = torch.tensor([-50.0, 50.0])
    y_coverage = torch.tensor([-10.0, 10.0])
    vehicle_line_thickness = 1
    vehicle_body_color = torch.tensor([255, 0, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    vehicle_head_color = torch.tensor([255, 255, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    dtype = torch.float32
    device = torch.device("cpu")
    
    lanes = torch.tensor([[-100.0, 0.0, 100.0, 0.0],
                            [-100.0, -4.0, 100.0, -4.0]])
    lane_width = 4.0
    lane_line_thickness = 1
    lane_line_color = (0, 255, 0)
    lane_boundary_line_thickness = 1
    lane_boundary_line_color = (255, 0, 255)
    
    brains = [lambda obs: torch.zeros(size=(obs.shape[0], 2))]
    brain_indexes = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    target_lanes = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1])
    speed = torch.tensor([10, 15, 20, 10, 10, 10, 10, 10, 10])
    time_interval = 0.05
    max_observed_distance = torch.zeros(x.shape[0]) + 100.0
    max_vehicle_speed = torch.tensor([100, 100, 100, 100, 100, 100, 100, 100, 100])
    min_vehicle_speed = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    max_vehicle_acceleration = torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 5])
    min_vehicle_acceleration = torch.tensor([-5, -5, -5, -5, -5, -5, -5, -5, -5])
    max_vehicle_steering_angle = torch.tensor([torch.pi/4, torch.pi/4, torch.pi/4, torch.pi/4, torch.pi/4, torch.pi/4,
                                            torch.pi/4, torch.pi/4, torch.pi/4])
    
    traffic_environment = TrafficEnvironment(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color,
                                            lanes, lane_width, lane_line_thickness, lane_line_color, lane_boundary_line_thickness, lane_boundary_line_color,
                                            brains, brain_indexes, target_lanes, speed, time_interval, max_observed_distance, max_vehicle_speed, min_vehicle_speed, max_vehicle_acceleration, min_vehicle_acceleration,
                                            max_vehicle_steering_angle, dtype, device, max_step_count=50)
    
    render_resolution = (300, 1500)
    speed_up_rate = 1
    video_writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 
                                  1/traffic_environment.time_interval*speed_up_rate, 
                                  (render_resolution[1], render_resolution[0]), isColor=True)
    
    traffic_environment.reset()
    for i in range(50):
        # canvas
        canvas = np.zeros((render_resolution[0], render_resolution[1], 3), dtype=np.uint8)
        
        # step
        action = torch.zeros(size=(2,))
        action[1] = -torch.pi/10
        obs, reward, done, info = traffic_environment.step(action)
        
        # render
        canvas = traffic_environment.render(canvas)
        cv2.imshow('test', canvas)
        cv2.waitKey(int(1000*traffic_environment.time_interval/speed_up_rate))
        
        # write
        video_writer.write(canvas)
        
        if done:
            continue
            break
        
    video_writer.release()
    
if __name__ == '__main__':
    test1()
    pass