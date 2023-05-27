import numpy as np

def generate_templates(h, gamma):
    # Calculate the dimensions of the template matrix
    template_h = round(h / 10) * 2 + 1
    template_w = round(h / 10) * 2 + 1

    # Create an empty template matrix filled with ones (background)
    template = (np.ones((template_h, template_w)), np.ones((template_h, template_w)))#, np.ones((template_h, template_w)), np.ones((template_h, template_w)))

    # Calculate the center coordinates of the circle
    center_x = template_w // 2
    center_y = round(h / 15)
    radius = round(h / 17)

    center = np.array(((center_x, center_y),(center_x, template_h-center_y)))

    # Calculate the line width
    line_width = round(h / 20)

    # Iterate over each pixel in the template matrix
    for i in range(template_h):
        for j in range(template_w):
            # Calculate the distance from the current pixel to the circle center
            distance = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
            # distance2 = np.sqrt((j - (center_x-radius)) ** 2 + (i - (center_y)) ** 2)
            # distance3 = np.sqrt((j - (center_x+radius)) ** 2 + (i - (center_y)) ** 2)

            # If the pixel is within the circle radius or under the center of the circle with the line width, set it to zero (template)
            if distance <= radius:
                template[1][i, j] = 0
                template[0][i, j] = 0

            # if distance2 <= radius:
            #     template[2][i, j] = 0
            # if distance3 <= radius:
            #     template[3][i, j] = 0
            # elif distance <= 2*radius:
            #     template[1][i, j] = distance/(2*radius)
                
            if (i > center_y and abs((j - center_x) - round(np.tan(np.radians(gamma)) * (i - center_y))) <= line_width // 2):
                template[0][i, j] = 0
                # template[2][i, j] = 0
                # template[3][i, j] = 0
            # if distance <= radius:
            #     template[i, j] = 0
            # elif (i > center_y and abs((j - center_x) - round(np.tan(np.radians(gamma)) * (i - center_y))) <= line_width):
            #     template[0][i, j] = abs((j - center_x) - round(np.tan(np.radians(gamma)) * (i - center_y)))/(1.4*line_width)
            # elif distance <= 2*radius:
            #     template[0][i, j] = distance/(2*radius)

    
    # Rotate the template by 180 degrees
    rotated_template = (np.rot90(template[0], 2), np.rot90(template[1], 2))#, np.rot90(template[2], 2), np.rot90(template[3], 2))
    
    return template, rotated_template, center