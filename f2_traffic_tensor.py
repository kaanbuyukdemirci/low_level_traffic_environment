import torch
import cv2
import numpy as np

from f1_angled_rectangle_tensor import AngledRectangleTensor

class TrafficTensor(AngledRectangleTensor):
    """A subclass of AngledRectangleTensor. It represents a traffic tensor. 
    It is used to calculate the position of the vehicles in a lane. Overall, it
    is used to calculate the lane-related state for the vehicles.
    
    Aside from the init parameters, it has the following attributes (m is the number of vehicles, n is the number of lane lines)):
        direction : torch.Tensor, shape=(m,)
            The direction of the vehicles. If the vehicles are to the left of the lane lines, then the direction is -1.
            If the vehicles are to the right of the lane lines, then the direction is 1.
        closest_distances : torch.Tensor, shape=(m,)
            The closest distances of the vehicles to the lane lines. 
            If the vehicles are to the left of the lane lines, then the distances are negative.
            If the vehicles are to the right of the lane lines, then the distances are positive.
        closest_lanes : torch.Tensor, shape=(m,)
            The closest lanes of the vehicles to the lane lines.
        lane_x : torch.Tensor, shape=(m,)
            The x coordinates of the vehicles after they are projected onto the closest lanes.
        lane_y : torch.Tensor, shape=(m,)
            The y coordinates of the vehicles after they are projected onto the closest lanes.
        lane_position : torch.Tensor, shape=(m,)
            The positions of the vehicles on their closest lanes.
        lane_positions : torch.Tensor, shape=(m, n)
            The positions of the vehicles on every lane.
        is_in_lane : torch.Tensor, shape=(m,)
            Whether the vehicles are in the lanes.
    """
    def __init__(self, x:torch.Tensor, y:torch.Tensor, width:torch.Tensor, length:torch.Tensor, angle:torch.Tensor, 
                 x_coverage:torch.Tensor, y_coverage:torch.Tensor, vehicle_line_thickness:float, 
                 vehicle_body_color:torch.Tensor, vehicle_head_color:torch.Tensor, 
                 lanes:torch.Tensor, lane_width:float, lane_line_thickness:float, lane_line_color:tuple[int,int,int], 
                 lane_boundary_line_thickness:float, lane_boundary_line_color:tuple[int,int,int],
                 dtype:torch.dtype, device:torch.device):
        """Constructor of TrafficTensor. Inherits from AngledRectangleTensor.

        Parameters (m is the number of vehicles, n is the number of lane lines)
        ----------
        x : torch.Tensor, shape=(m,)
            The x coordinates of the centers of the vehicles (rectangles).
        y : torch.Tensor, shape=(m,)
            The y coordinates of the centers of the vehicles (rectangles).
        width : torch.Tensor, shape=(m,)
            The width of the vehicles (rectangles).
        length : torch.Tensor, shape=(m,)
            The length of the vehicles (rectangles).
        angle : torch.Tensor, shape=(m,)
            The angle of the vehicles (rectangles).
        x_coverage : torch.Tensor, shape=(2,)
            The x coverage of the image.
        y_coverage : torch.Tensor, shape=(2,)
            The y coverage of the image.
        vehicle_line_thickness : float
            The line thickness of the vehicles (rectangles).
        vehicle_body_color : torch.Tensor, shape=(m,3)
            The color of the bodies of the vehicles (rectangles).
        vehicle_head_color : torch.Tensor, shape=(m,3)
            The color of the heads of the vehicles (rectangles).
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
            The width of the lane. If a vehicle is within this distance to a lane line,
            then it is considered to be in the lane.
        lane_line_thickness : float
            The line thickness of the lane lines.
        lane_line_color : tuple[int,int,int]
            The color of the lane lines.
        lane_boundary_line_thickness : float
            The line thickness of the lane boundary lines.
        lane_boundary_line_color : tuple[int,int,int]
            The color of the lane boundary lines.
        dtype : torch.dtype
            The data type of the tensors.
        device : torch.device
            The device of the tensors.
        """
        super().__init__(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color, dtype, device)
        self.lanes = lanes.to(dtype=dtype, device=device)
        self.lane_width = lane_width
        self.lane_line_thickness = lane_line_thickness
        self.lane_line_color = lane_line_color
        self.lane_boundary_line_thickness = lane_boundary_line_thickness
        self.lane_boundary_line_color = lane_boundary_line_color
        
        self.closest_distances = None
        self.closest_lanes = None
        self.lane_x = None
        self.lane_y = None
        self.lane_position = None
        self.is_in_lane = None
        self.lane_positions = None

    def update(self):
        """Update the attributes of the traffic tensor. This method should be called
        every time one of the init parameters is updated.
        """
        super().update()
        self.update_closest()
        self.update_if_in_lane()

    def update_closest(self) -> torch.Tensor:
        """Get the distance from a point to the closest lane line. Also get the closest lane line.
        """
        # m is the number of vehicles, n is the number of lane lines
        
        # project the point to each lane line
        projection_vectors = self.lanes[:, 2:] - self.lanes[:, :2] # shape = (n, 2)
        projection_vectors = projection_vectors / torch.norm(projection_vectors, dim=1, keepdim=True) # shape = (n, 2)
        projection_vectors = projection_vectors.unsqueeze(0) # shape = (1, n, 2)
        
        # calculate the projected point on each lane line
        points = torch.stack((self.x, self.y), dim=1) # shape = (m, 2)
        points = points.unsqueeze(1) # shape = (m, 1, 2)
        point_vectors = points - self.lanes[:, :2].unsqueeze(0) # shape = (m, n, 2)
        projection_lengths = torch.sum(point_vectors * projection_vectors, dim=2) # shape = (m, n)
        projection_lengths = projection_lengths.unsqueeze(2) # shape = (m, n, 1)
        max_lengths = torch.norm(self.lanes[:, 2:] - self.lanes[:, :2], dim=1, keepdim=True) # shape = (n, 1)
        min_lengths = torch.zeros_like(max_lengths) # shape = (n, 1)
        projection_lengths = torch.min(projection_lengths, max_lengths) # shape = (m, n, 1)
        projection_lengths = torch.max(projection_lengths, min_lengths) # shape = (m, n, 1)
        projected_points = self.lanes[:, :2].unsqueeze(0) + projection_lengths * projection_vectors # shape = (m, n, 2)
        
        # calculate the distance from the point to each lane line
        distances = torch.norm(points - projected_points, dim=2) # shape = (m, n)
        closest_distances, closest_lanes = torch.min(distances, dim=1)
        
        # calculate the angle of every vehicle with respect to their closest lane
        lane_vector = projection_vectors.squeeze(0)[closest_lanes] # shape = (m, 2)
        lane_angle = torch.atan2(lane_vector[:, 1], lane_vector[:, 0]) # shape = (m,)
        
        lane_start_point = self.lanes[closest_lanes, :2] # shape = (m, 2)
        vehicle_vector = points.squeeze(1) - lane_start_point # shape = (m, 2)
        vehicle_angle = torch.atan2(vehicle_vector[:, 1], vehicle_vector[:, 0]) # shape = (m,)
        
        relative_angle = (vehicle_angle - lane_angle) # shape = (m,)
        relative_angle = torch.where(relative_angle > torch.pi, relative_angle - 2*torch.pi, relative_angle)
        direction = torch.where(relative_angle > 0, 1, -1)
        
        # update self values
        self.direction = direction
        self.closest_distances = closest_distances * direction
        self.closest_lanes = closest_lanes
        self.lane_x = projected_points[torch.arange(0,projected_points.shape[0]), closest_lanes, 0]
        self.lane_y = projected_points[torch.arange(0,projected_points.shape[0]), closest_lanes, 1]
        self.lane_position = projection_lengths[torch.arange(0,projection_lengths.shape[0]), closest_lanes, 0]
        self.lane_positions = projection_lengths.squeeze(2) # shape = (m, n)
    
    def update_if_in_lane(self) -> torch.Tensor:
        """Check if the vehicles are in lane.
        """
        distances = self.closest_distances
        self.is_in_lane = torch.abs(distances) < (self.lane_width / 2)
    
    def render(self, image:np.ndarray, mirror=True) -> np.ndarray:
        """Render the lanes on the image.

        Parameters
        ----------
        image : np.ndarray, shape=(y_res, x_res, 3), dtype=np.uint8
            The image to be rendered on.
        mirror : bool, optional
            Whether to mirror the image in y-axis so that indexing is same as a usual cartesian coordinate system, by default True
            
        Returns
        -------
        np.ndarray, shape=(y_res, x_res, 3), dtype=np.uint8
            The rendered image. The renders are drawn on the given image. So the returned image is the same as the given image. 
            So it is not necessary to assign the returned image to a variable.
        """
        # shift and scale the lane lines to the image
        shifter = torch.tensor([self.x_coverage[0], 
                                self.y_coverage[0], 
                                self.x_coverage[0], 
                                self.y_coverage[0]], dtype=torch.float32).view(1, -1)
        coverage_scaler = torch.tensor([self.x_coverage[1] - self.x_coverage[0], 
                                        self.y_coverage[1] - self.y_coverage[0], 
                                        self.x_coverage[1] - self.x_coverage[0], 
                                        self.y_coverage[1] - self.y_coverage[0]], dtype=torch.float32).view(1, -1)
        image_scaler = torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]], dtype=torch.float32).view(1, -1)
        
        points = self.lanes - shifter
        points = points / coverage_scaler
        points = points * image_scaler
        
        up_points = self.lanes + torch.tensor([-self.lane_width, +self.lane_width, +self.lane_width, +self.lane_width]).view(1,-1) / 2
        up_points = up_points - shifter
        up_points = up_points / coverage_scaler
        up_points = up_points * image_scaler
        
        down_points = self.lanes + torch.tensor([-self.lane_width, -self.lane_width, +self.lane_width, -self.lane_width]).view(1,-1) / 2
        down_points = down_points - shifter
        down_points = down_points / coverage_scaler
        down_points = down_points * image_scaler
        
        vehicle_lane_points = torch.stack((self.lane_x, self.lane_y), dim=1) # shape = (m, 2)
        vehicle_lane_points = vehicle_lane_points - shifter[:, :2]
        vehicle_lane_points = vehicle_lane_points / coverage_scaler[:, :2]
        vehicle_lane_points = vehicle_lane_points * image_scaler[:, :2]
        
        if mirror:
            image = cv2.flip(image, 0)
        
        super().render(image, False)
        
        # render the lane boundary lines
        for up_point, down_point in zip(up_points, down_points):
            up_point = up_point.numpy().astype(np.int32)
            down_point = down_point.numpy().astype(np.int32)
            cv2.line(image, (up_point[0], up_point[1]), (up_point[2], up_point[3]), self.lane_boundary_line_color, self.lane_line_thickness)
            cv2.line(image, (down_point[0], down_point[1]), (down_point[2], down_point[3]), self.lane_boundary_line_color, self.lane_line_thickness)
        
        # render the lane lines
        for point in points:
            point = point.numpy().astype(np.int32)
            cv2.line(image, (point[0], point[1]), (point[2], point[3]), self.lane_line_color, self.lane_line_thickness)
        
        # render the positions on the lanes
        for i, vehicle_lane_point in enumerate(vehicle_lane_points):
            if self.is_in_lane[i]:
                head_color = self.head_color[i].tolist()
                vehicle_lane_point = vehicle_lane_point.numpy().astype(np.int32)
                cv2.circle(image, (vehicle_lane_point[0], vehicle_lane_point[1]), self.line_thickness*3, head_color, -1)
        
        if mirror:
            image = cv2.flip(image, 0)
        
        return image

    def __str__(self):
        return str(vars(self))

def test():
    x = torch.tensor([0.0, 5.0, 10.0])
    y = torch.tensor([4.9, 5.1, -9.0])
    width = torch.tensor([1.0, 5.0, 10.0])
    length = torch.tensor([1.0, 5.0, 20.0])
    angle = torch.tensor([0.0, 0.0, -torch.pi/4])
    x_coverage = torch.tensor([-100.0, 100.0])
    y_coverage = torch.tensor([-20.0, 20.0])
    vehicle_line_thickness = 1
    vehicle_body_color = torch.tensor([255, 0, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    vehicle_head_color = torch.tensor([255, 255, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    dtype = torch.float32
    device = torch.device("cpu")
    
    lanes = torch.tensor([[0.0, -10.0, 10.0, -10.0],
                            [0.0, 10.0, 10.0, 10.0]])
    lane_width = 10.0
    lane_line_thickness = 1
    lane_line_color = (0, 255, 0)
    lane_boundary_line_thickness = 1
    lane_boundary_line_color = (255, 0, 255)
    
    traffic_tensor = TrafficTensor(x, y, width, length, angle, x_coverage, y_coverage, vehicle_line_thickness, vehicle_body_color, vehicle_head_color, 
                                    lanes, lane_width, lane_line_thickness, lane_line_color, lane_boundary_line_thickness, lane_boundary_line_color,
                                    dtype, device)
    
    traffic_tensor.update()
    print(traffic_tensor)
    if False:
        dic = vars(traffic_tensor)
        print(dic["intersection_matrix"])
        print(dic["intersection_vector"])
        print(dic["x"])
        print(dic["y"])
    
    canvas = np.zeros((300, 1500, 3), dtype=np.uint8)
    canvas = traffic_tensor.render(canvas)
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)

if __name__ == "__main__":
    test()