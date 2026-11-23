import random
import numpy as np
import cv2


def generate_image(height=256, width=256, number_triangles=5, number_quadrilaterals=5, number_ellipses=5, number_stars=5):
    def draw_triangle(img):
        while True:
            points = np.array([[random.randint(-40, width + 40), random.randint(-40, height + 40)],
                               [random.randint(-40, width + 40), random.randint(-40, height + 40)],
                               [random.randint(-40, width + 40), random.randint(-40, height + 40)]], dtype=np.int32)
    
            if np.any((0 <= points[:, 0]) & (points[:, 0] < width) & (0 <= points[:, 1]) & (points[:, 1] < height)):
                break  # At least one point is inside the canvas, exit the loop

        points = points.reshape((-1, 1, 2))
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillPoly(img, [points], color=random_color, lineType=cv2.LINE_AA)

    def draw_quadrilateral(img):
        while True:
            points = np.array([[random.randint(-40, width + 40), random.randint(-40, height + 40)],
                               [random.randint(-40, width + 40), random.randint(-40, height + 40)],
                               [random.randint(-40, width + 40), random.randint(-40, height + 40)],
                               [random.randint(-40, width + 40), random.randint(-40, height + 40)]], dtype=np.int32)
        
            if np.any((0 <= points[:, 0]) & (points[:, 0] < width) & (0 <= points[:, 1]) & (points[:, 1] < height)):
                break  # At least one point is inside the canvas, exit the loop

        points = points.reshape((-1, 1, 2))
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillPoly(img, [points], color=random_color, lineType=cv2.LINE_AA)

    def draw_ellipse(img):
        while True:
            center = (random.randint(-40, width + 40), random.randint(-40, height + 40))
            axes = (random.randint(50, 150), random.randint(50, 150))
            angle = random.randint(0, 360)
            # Check if at least part of the ellipse is inside the canvas
            if (0 <= center[0] <= width and 0 <= center[1] <= height) or \
               (0 <= center[0] + axes[0] <= width and 0 <= center[1] <= height) or \
               (0 <= center[0] - axes[0] <= width and 0 <= center[1] <= height) or \
               (0 <= center[0] <= width and 0 <= center[1] + axes[1] <= height) or \
               (0 <= center[0] <= width and 0 <= center[1] - axes[1] <= height):
                break
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.ellipse(img, center, axes, angle, 0, 360, random_color, -1, lineType=cv2.LINE_AA)

    def draw_star(img):
        center = (random.randint(1, width - 1), random.randint(1, height - 1))
        points = []
        for _ in range(5):
            x = random.randint(center[0] - 100, center[0] + 100)
            y = random.randint(center[1] - 100, center[1] + 100)
            points.append((x,y))
        
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for point in points:
            cv2.line(img, center, point, random_color, thickness=1, lineType=cv2.LINE_AA)
    
    # Create one color canvas
    img = np.ones((height, width, 3), dtype=np.uint8)
    random_color = np.random.randint(1, 256, (3,), dtype=np.uint8)
    img[:, :, 0] = random_color[0]
    img[:, :, 1] = random_color[1]
    img[:, :, 2] = random_color[2]

    shapes = []
    shapes.extend([1] * number_triangles)          # 1 for triangles
    shapes.extend([2] * number_quadrilaterals)     # 2 for quadrilaterals
    shapes.extend([3] * number_ellipses)           # 4 for ellipses
    shapes.extend([4] * number_stars)              # 3 for stars
    np.random.shuffle(shapes)
    
    for shape_id in shapes:
        if shape_id == 1:
            draw_triangle(img)
        elif shape_id == 2:
            draw_quadrilateral(img)
        elif shape_id == 3:
            draw_ellipse(img)
        elif shape_id == 4:
            draw_star(img)
    return img


