import torch
import cv2
import numpy as np

class AngledRectangleTensor(object):
    """An object that stores a tensor of angled rectangles, 
    and provides methods to calculate the intersection of these rectangles, 
    and draw the rectangles on an image. Overall, it handles the geometry of the rectangles.
    
    Aside from init parameters, you can also access the following attributes (m is the number of rectangles):
        point_1 : torch.Tensor, shape (m, 2)
            The first corner point of the rectangles.
        point_2 : torch.Tensor, shape (m, 2)
            The second corner point of the rectangles.
        point_3 : torch.Tensor, shape (m, 2)
            The third corner point of the rectangles.
        point_4 : torch.Tensor, shape (m, 2)
            The fourth corner point of the rectangles.
        intersection_matrix : torch.Tensor, shape (m, m)
            The intersection matrix of the rectangles, 
            where intersection_matrix[i, j] is 1 if rectangle i and rectangle j intersect, and 0 otherwise.
        intersection_vector : torch.Tensor, shape (m, )
            The intersection vector of the rectangles,
            where intersection_vector[i] is 1 if rectangle i intersects with any other rectangle, and 0 otherwise.
    """
    def __init__(self, x:torch.Tensor, y:torch.Tensor, width:torch.Tensor, length:torch.Tensor, angle:torch.Tensor, 
                 x_coverage:torch.Tensor, y_coverage:torch.Tensor, line_thickness:float, body_color:torch.Tensor, 
                 head_color:torch.Tensor, dtype:torch.dtype, device:torch.device):
        """Construct an AngledRectangleTensor object.

        Parameters (m is the number of rectangles)
        ----------
        x : torch.Tensor, shape (m, )
            The x coordinates of the centers of the rectangles.
        y : torch.Tensor, shape (m, )
            The y coordinates of the centers of the rectangles.
        width : torch.Tensor, shape (m, )
            The width of the rectangles.
        length : torch.Tensor, shape (m, )
            The length of the rectangles.
        angle : torch.Tensor, shape (m, )
            The angle of the rectangles.
        x_coverage : torch.Tensor, shape (2, )
            The x coverage of the image.
        y_coverage : torch.Tensor, shape (2, )
            The y coverage of the image.
        line_thickness : float
            The line thickness of the rectangles.
        body_color : torch.Tensor, shape (m, 3)
            The body color of the rectangles. (RGB)
        head_color : torch.Tensor, shape (m, 3)
            The head color of the rectangles. (RGB)
        dtype : torch.dtype
            The data type of the tensors.
        device : torch.device
            The device of the tensors.
        """
        
        self.x = x.to(dtype=dtype, device=device)
        self.y = y.to(dtype=dtype, device=device)
        self.width = width.to(dtype=dtype, device=device)
        self.length = length.to(dtype=dtype, device=device)
        self.angle = angle.to(dtype=dtype, device=device)
        self.x_coverage = x_coverage
        self.y_coverage = y_coverage
        self.line_thickness = line_thickness
        self.body_color = body_color
        self.head_color = head_color
        self.dtype = dtype
        self.device = device
        
        self.point_1 = None
        self.point_2 = None
        self.point_3 = None
        self.point_4 = None
        self.intersection_matrix = None
        self.intersection_vector = None
    
    def update(self):
        """Update the attributes of the angled rectangles. This method should be called
        every time one of the init parameters is updated.
        """
        self.update_points()
        self.update_intersection_info()
    
    def update_points(self):
        """Update the corner points of the rectangles, based on the centers, width, length, and angle of the rectangles.
        """
        cos_constant = torch.cos(self.angle) * 0.5
        sin_constant = torch.sin(self.angle) * 0.5
        self.point_1 = torch.stack(((self.x - sin_constant * self.width - cos_constant * self.length),
                                  (self.y + cos_constant * self.width - sin_constant * self.length)), dim=1)
        self.point_2 = torch.stack(((self.x + sin_constant * self.width - cos_constant * self.length),
                                  (self.y - cos_constant * self.width - sin_constant * self.length)), dim=1)
        self.point_3 = torch.stack(((2 * self.x - self.point_1[:,0]),
                                  (2 * self.y - self.point_1[:,1])), dim=1)
        self.point_4 = torch.stack(((2 * self.x - self.point_2[:,0]),
                                  (2 * self.y - self.point_2[:,1])), dim=1)
        
    def update_intersection_info(self):
        """Update the intersection matrix and intersection vector to see whether the rectangles intersect,
        based on the corner points of the rectangles.
        """
        # self.point_1 is a tensor of shape (m, 2), which stores the coordinates of the first point of each rectangle
        # self.point_2 is a tensor of shape (m, 2), which stores the coordinates of the second point of each rectangle
        # self.point_3 is a tensor of shape (m, 2), which stores the coordinates of the third point of each rectangle
        # self.point_4 is a tensor of shape (m, 2), which stores the coordinates of the fourth point of each rectangle
        
        # separating axis theorem
        
        # number of rectangles
        n_rectangles = self.point_1.shape[0]
        
        # calculate the axes of the rectangles, and normalize them
        axes = torch.stack(((self.point_2 - self.point_1), 
                            (self.point_3 - self.point_2), 
                            (self.point_4 - self.point_3), 
                            (self.point_1 - self.point_4)), dim=1)
        axes = axes / torch.norm(axes, dim=2, keepdim=True)
        axes = axes.view(n_rectangles, 4, 1, 1, 2)
        # shape of axes: (n_rectangles, 4, 1, 1, 2) # a 2D vector for each 4 axes per rectangle.
        
        # calculate the projections of the rectangles on the axes
        point_1_projections = torch.matmul(axes, self.point_1.view(1, 1, n_rectangles, 2, 1)).view(n_rectangles, 4, n_rectangles)
        point_2_projections = torch.matmul(axes, self.point_2.view(1, 1, n_rectangles, 2, 1)).view(n_rectangles, 4, n_rectangles)
        point_3_projections = torch.matmul(axes, self.point_3.view(1, 1, n_rectangles, 2, 1)).view(n_rectangles, 4, n_rectangles)
        point_4_projections = torch.matmul(axes, self.point_4.view(1, 1, n_rectangles, 2, 1)).view(n_rectangles, 4, n_rectangles)
        projections = torch.stack((point_1_projections, point_2_projections, point_3_projections, point_4_projections), dim=3)
        # shape of projections: (n_rectangles, 4, n_rectangles, 4)
        # projections[i,j,k,l] means:
        #  - ith rectangle
        #  - jth axis of the ith rectangle
        #  - kth rectangle
        #  - lth point of the kth rectangle
        # the projection of the lth point of the kth rectangle on the jth axis of the ith rectangle
        
        # calculate the minimum and maximum projections of the rectangles on the axes to detect intersections
        min_projections = torch.min(projections, dim=3, keepdim=False)[0]
        max_projections = torch.max(projections, dim=3, keepdim=False)[0]
        # shape of min_projections: (n_rectangles, 4, n_rectangles)
        # shape of max_projections: (n_rectangles, 4, n_rectangles)
        
        # diagonals
        min_projections_diagonal = torch.diagonal(min_projections, dim1=0, dim2=2).transpose(0,1).unsqueeze(2)
        max_projections_diagonal = torch.diagonal(max_projections, dim1=0, dim2=2).transpose(0,1).unsqueeze(2)
        # shape of min_projections_diagonal: (n_rectangles, 4, 1)
        # shape of max_projections_diagonal: (n_rectangles, 4, 1)
        
        # calculate the intersection matrix
        # i'th and j'th rectangles doesn't intersect if and only if:
        # (max_projections[i,k,i] < min_projections[i,k,j]) or (max_projections[i,k,j] < min_projections[i,k,i]) for any k
        # min_projections[i,k,j] is obtained by min_projections
        # min_projections[i,k,i] is obtained by min_projections_diagonal
        # max_projections[i,k,j] is obtained by max_projections
        # max_projections[i,k,i] is obtained by max_projections_diagonal
        intersection_matrix = (max_projections_diagonal < min_projections) | (max_projections < min_projections_diagonal)
        intersection_matrix_self = torch.all(~intersection_matrix, dim=1)
        intersection_matrix_other = intersection_matrix_self.transpose(0,1)
        intersection_matrix = intersection_matrix_self & intersection_matrix_other
        # shape of intersection_matrix: (n_rectangles, n_rectangles)
        
        # calculate the intersection vector
        intersection_vector = torch.sum(intersection_matrix, dim=1) > 1
        
        self.intersection_matrix = intersection_matrix
        self.intersection_vector = intersection_vector
    
    def render(self, image:np.ndarray, mirror=True) -> np.ndarray:
        """Render the rectangles on the image.

        Parameters
        ----------
        image : np.ndarray, shape=(y_res, x_res, 3), dtype=np.uint8
            The image to render on.
        mirror : bool, optional
            Whether to mirror the image in y-axis so that indexing is same as a usual cartesian coordinate system, by default True

        Returns
        -------
        np.ndarray, shape=(y_res, x_res, 3), dtype=np.uint8
            The rendered image.
        """
        # shift and scale
        y_res, x_res = image.shape[0:2]
        point_10 = (((self.point_1[:,0] - self.x_coverage[0]) / (self.x_coverage[1] - self.x_coverage[0])) * x_res).int()
        point_11 = (((self.point_1[:,1] - self.y_coverage[0]) / (self.y_coverage[1] - self.y_coverage[0])) * y_res).int()
        point_20 = (((self.point_2[:,0] - self.x_coverage[0]) / (self.x_coverage[1] - self.x_coverage[0])) * x_res).int()
        point_21 = (((self.point_2[:,1] - self.y_coverage[0]) / (self.y_coverage[1] - self.y_coverage[0])) * y_res).int()
        point_30 = (((self.point_3[:,0] - self.x_coverage[0]) / (self.x_coverage[1] - self.x_coverage[0])) * x_res).int()
        point_31 = (((self.point_3[:,1] - self.y_coverage[0]) / (self.y_coverage[1] - self.y_coverage[0])) * y_res).int()
        point_40 = (((self.point_4[:,0] - self.x_coverage[0]) / (self.x_coverage[1] - self.x_coverage[0])) * x_res).int()
        point_41 = (((self.point_4[:,1] - self.y_coverage[0]) / (self.y_coverage[1] - self.y_coverage[0])) * y_res).int()
        
        # mirror if necessary
        if mirror:
            image = cv2.flip(image, 0)
        
        # render each rectangle one by one
        for rectangle_index in range(point_10.shape[0]):
            body_color = self.body_color[rectangle_index].tolist()
            head_color = self.head_color[rectangle_index].tolist()
            cv2.line(image, 
                     (point_10[rectangle_index].item(), point_11[rectangle_index].item()), 
                     (point_20[rectangle_index].item(), point_21[rectangle_index].item()), 
                     body_color, self.line_thickness, cv2.LINE_AA)
            cv2.line(image, 
                     (point_20[rectangle_index].item(), point_21[rectangle_index].item()), 
                     (point_30[rectangle_index].item(), point_31[rectangle_index].item()), 
                     body_color, self.line_thickness, cv2.LINE_AA)
            cv2.line(image, 
                     (point_30[rectangle_index].item(), point_31[rectangle_index].item()), 
                     (point_40[rectangle_index].item(), point_41[rectangle_index].item()), 
                     head_color, self.line_thickness, cv2.LINE_AA)
            cv2.line(image, 
                     (point_40[rectangle_index].item(), point_41[rectangle_index].item()), 
                     (point_10[rectangle_index].item(), point_11[rectangle_index].item()), 
                     body_color, self.line_thickness, cv2.LINE_AA)

        # print the rectangle index on the image at the center of the rectangle
        for rectangle_index in range(point_10.shape[0]):
            head_color = self.head_color[rectangle_index].tolist()
            width = np.sqrt((point_21[rectangle_index] - point_11[rectangle_index])**2 + (point_20[rectangle_index] - point_10[rectangle_index])**2)
            scale = float(abs(width / 40))
            if rectangle_index < 10:
                x_org = int((point_10[rectangle_index].item() + point_30[rectangle_index].item()) / 2 - 10 * scale)
            elif rectangle_index > 9 and rectangle_index < 100:
                x_org = int((point_10[rectangle_index].item() + point_30[rectangle_index].item()) / 2 - 20 * scale)
            elif rectangle_index > 99:
                x_org = int((point_10[rectangle_index].item() + point_30[rectangle_index].item()) / 2 - 30 * scale)
            y_org = int((point_11[rectangle_index].item() + point_31[rectangle_index].item()) / 2 - 10 * scale)
            cv2.putText(img=image, 
                        text=str(rectangle_index), 
                        org=(x_org, y_org),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=scale, 
                        color=head_color, 
                        thickness=self.line_thickness,
                        lineType=cv2.LINE_AA,
                        bottomLeftOrigin=True)
        
        # mirror if necessary
        if mirror:
            image = cv2.flip(image, 0)
        
        return image
    
    def __str__(self):
        return str(vars(self))
    
def test():
    x = torch.tensor([0.0, 0.0, 10.0])
    y = torch.tensor([0.0, 0.0, -10.0])
    width = torch.tensor([1.0, 10.0, 10.0])
    length = torch.tensor([1.0, 10.0, 20.0])
    angle = torch.tensor([0.0, 0.0, torch.pi/4])
    x_coverage = torch.tensor([-50.0, 50.0])
    y_coverage = torch.tensor([-10.0, 10.0])
    line_thickness = 1
    body_color = torch.tensor([255, 0, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    head_color = torch.tensor([255, 255, 0]).unsqueeze(0) * torch.ones(x.shape[0]).unsqueeze(1)
    dtype = torch.float32
    device = torch.device("cpu")
    
    angled_rectangle_tensor = AngledRectangleTensor(x, y, width, length, angle, x_coverage, y_coverage, 
                                                    line_thickness, body_color, head_color, dtype, device)
    
    angled_rectangle_tensor.update()
    print(angled_rectangle_tensor)
    if False:
        dic = vars(angled_rectangle_tensor)
        print(dic["intersection_matrix"])
        print(dic["intersection_vector"])
    
    canvas = np.zeros((300, 1500, 3), dtype=np.uint8)
    canvas = angled_rectangle_tensor.render(canvas)
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)

if __name__ == "__main__":
    test()