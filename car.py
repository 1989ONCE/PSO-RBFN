import math
import numpy as np
import matplotlib.pyplot as plt

class Car():
    def __init__(self, initX, initY, phi, track):
        self.radius = 3             # the radius of the car
        self.currentX = initX       # the current X coordinate of the car
        self.currentY = initY       # the current Y coordinate of the car
        self.currentPHI = phi        # φ: the angle between the car and the horizontal axis
        self.currentTHETA = 0       # θ: the angle that the steering wheel rotates
        
        # Sensor distances
        self.front_distance = 22.0   
        self.left_distance = 8.4853    
        self.right_distance = 8.4853  

        self.track = track

    def update_position(self):
        self.currentX = self.currentX + math.cos(math.radians(self.currentPHI + self.currentTHETA)) + (math.sin(math.radians(self.currentTHETA)) * math.sin(math.radians(self.currentPHI)))
        self.currentY = self.currentY + math.sin(math.radians(self.currentPHI + self.currentTHETA)) - (math.sin(math.radians(self.currentTHETA)) * math.cos(math.radians(self.currentPHI)))
        self.currentPHI = math.degrees(math.radians(self.currentPHI) - math.asin((2 * math.sin(math.radians(self.currentTHETA))) / (2 * self.radius)))
        
        distances = self.calculate_sensors(self.currentX, self.currentY, self.currentPHI)
        self.front_distance = distances[0]
        self.left_distance = distances[1]
        self.right_distance = distances[2]

    def draw_sensor_distance_arrow(self, ax, direction, center_x, center_y, phi, distance):
         # Color for the sensor lines
        if direction == 'Front':
            color = 'red'
        elif direction == 'Right':
            color = 'blue'
        else:
            color = 'green'
    
        # Convert orientation phi from degrees to radians
        phi_radians = math.radians(phi)
        distance = distance - 2
        arrow = ax.arrow(center_x, center_y, distance * math.cos(phi_radians), distance * math.sin(phi_radians), head_width=1, head_length=2, fc=color, ec=color)
        return arrow

    def draw_car(self, ax):
        car = plt.Circle((self.currentX, self.currentY), self.radius, edgecolor='black', facecolor='none')
        center_point = plt.Circle((self.currentX, self.currentY), 0.5, color='darkgrey')
        ax.add_artist(center_point)
        ax.add_artist(car)
        text = ax.text(0.95, 0.05, f'Car Center: ({self.currentX:.2f}, {self.currentY:.2f})\n Front: {self.front_distance:.2f}\n Left: {self.left_distance:.2f}\n Right: {self.right_distance:.2f}', transform=ax.transAxes, fontsize=8, ha='right', va='bottom')
        
        return car, text, center_point
    
    # 計算感測器到牆壁的距離
    def calculate_sensors(self, x, y, angle):
        angle_rad = np.radians(angle)
        
        # 感測器角度(相對於車子方向)
        angles = [0, 45, -45]  # front, left, right
        distances = []
        
        for sensor_angle in angles:
            # 計算感測器絕對角度
            abs_angle = angle_rad + np.radians(sensor_angle)
            
            # 創建感測器射線
            max_distance = 50
            ray_points = np.array([
                [x, y],
                [x + max_distance * np.cos(abs_angle),
                y + max_distance * np.sin(abs_angle)]
            ])
            
            min_dist = max_distance
            
            # 檢查射線與每個牆段是否相交
            for i in range(len(self.track) - 1):
                wall = np.array([self.track[i], self.track[i + 1]])
                
                # 線段相交計算
                x1, y1 = ray_points[0]
                x2, y2 = ray_points[1]
                x3, y3 = wall[0]
                x4, y4 = wall[1]
                
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator == 0:  # 若平行則跳過
                    continue
                    
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                
                if 0 <= t <= 1 and 0 <= u <= 1:  # 若相交則計算距離
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_y = y1 + t * (y2 - y1)
                    dist = np.sqrt((x - intersection_x)**2 + (y - intersection_y)**2)
                    min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
        
        return distances
    
    def check_collision(self):
        # Iterate over each line segment in the track
        for i in range(len(self.track) - 1):
            start = np.array(self.track[i])
            end = np.array(self.track[i + 1])
            
            # Check for intersection between the circle and the line segment
            if self.circle_line_segment_intersect(start, end, np.array([self.currentX, self.currentY]), self.radius):
                return True  # Collision detected
        
        return False  # No collision detected
    
    def circle_line_segment_intersect(self, pt1, pt2, center, radius):
        # Vector from pt1 to pt2
        d = pt2 - pt1
        # Vector from pt1 to circle center
        f = pt1 - center
    
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
    
        discriminant = b**2 - 4 * a * c
    
        if discriminant < 0:
            # No intersection
            return False
        else:
            # There may be an intersection
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
    
            # Check if either t1 or t2 is within the segment [0,1]
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True  # Intersection occurs within the segment
            else:
                return False  # Intersection occurs outside the segment
            
    def set_currentTHETA(self, theta):
        self.currentTHETA = theta
    
    def get_distances(self):
        return self.front_distance, self.left_distance, self.right_distance