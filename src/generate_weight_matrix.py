import numpy as np

def create_array(wh, upper_limit, lower_limit, center, slope):
    # Create an array of zeros
    array = np.zeros(wh)

    # Calculate the values using the decreasing sigmoid function
    x = np.linspace(0, 1, wh)
    array = upper_limit - (upper_limit - lower_limit) / (1 + np.exp(-slope * (x - center)))

    return array

def generate_matrix(wh, ww, wtha, wthb, upper_limit, lower_limit, center_correction, slope):
    # Compute the number of columns
    num_columns = ww

    # Compute the linearly spaced centers
    centers = np.linspace(wtha, wthb, num_columns)

    # Create an empty matrix
    matrix = np.zeros((wh, ww))

    # Generate each column using create_array function
    for i, center in enumerate(centers):
        matrix[:, i] = create_array(wh, upper_limit, lower_limit, center + center_correction, slope)
        if upper_limit<lower_limit:
            matrix[round(center*wh):, i] += create_array(round((1-center)*wh), 0, -2, 0.85, slope)
        else:
            matrix[:round(center*wh), i] += create_array(round(center*wh), -2, 0, 0.15, slope)

    return matrix