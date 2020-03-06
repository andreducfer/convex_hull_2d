import numpy as np
import matplotlib.pyplot as plt

points = np.loadtxt("dataset/fecho1.txt").astype(np.float)

def get_index_to_start_hull(points):
    min_y = np.argmin(points[:,1])
    index_to_use = min_y
    index_elements_to_consider = []

    for i, element in enumerate(points):
        if element[1] == points[min_y, 1]:
            index_elements_to_consider.append(i)

    max_sum = 0
    for index in index_elements_to_consider:
        current_sum = abs(points[index, 0] - points[index, 1])

        if current_sum > max_sum:
            index_to_use = index
            max_sum = current_sum

    return index_to_use

def get_angle_of_points(points, index_to_start_hull):
    array_angles = np.zeros(len(points))

    start_point = points[index_to_start_hull]

    for i, current_point in enumerate(points):
        if i == index_to_start_hull:
            array_angles[i] = 0
            continue
        
        diff_point = (current_point - start_point)
        p = (diff_point) / np.sqrt(np.dot(diff_point, diff_point))
        array_angles[i] = np.arccos(np.dot(np.array([1, 0]), p))

    return array_angles

def convex_angle(point_1, point_2, point_3):
    crossp = ((point_2[0] - point_1[0]) * (point_3[1] - point_1[1]) -
              (point_2[1] - point_1[1]) * (point_3[0] - point_1[0]))
    return crossp > 0

def graham_scan(points, original_indexes):
    hull = [original_indexes[0], original_indexes[1]]

    for i in range(2, len(points)):
        if convex_angle(points[hull[-2]], points[hull[-1]], points[original_indexes[i]]):
            hull.append(original_indexes[i])
        else:
            while len(hull) > 1 and not convex_angle(points[hull[-2]], points[hull[-1]], points[original_indexes[i]]):
                hull.pop()

            hull.append(original_indexes[i])

    return hull

def convex_hull(points):
    original_indexes = np.arange(len(points))

    # Get the index with small y. If there is a tie returns the index of the smallest y value with largest x value
    index_to_start_hull = get_index_to_start_hull(points)

    # Get list of angles of each point regarding to the point chosen to start the hull
    angle_of_points = get_angle_of_points(points, index_to_start_hull)
    
    # Sort list of indexes acording to the angles of all points regarding to the point used to start the hull
    index_to_sort_points = np.argsort(angle_of_points)
    original_indexes = original_indexes[index_to_sort_points]

    # Call function Graham Scan to find the convex hull
    convex_hull_indexes = graham_scan(points, original_indexes)

    return convex_hull_indexes

def print_solution(file_name, convex_hull_indexes):       
    with open(file_name, mode='w+') as fp:
        for index in convex_hull_indexes:
            print(str(index), file=fp)
        fp.close()

def draw_polygon(points, convex_hull_indexes, datasetname):
    x_polygon = points[:,0]
    y_polygon = points[:,1]

    plt.scatter(x_polygon, y_polygon, color="black", marker='.')

    for i in range(len(convex_hull_indexes)):
        x_hull = np.array([points[convex_hull_indexes[i - 1]][0], points[convex_hull_indexes[i]][0]])
        y_hull = np.array([points[convex_hull_indexes[i - 1]][1], points[convex_hull_indexes[i]][1]])
        plt.plot(x_hull[:], y_hull[:], linewidth=1, color="green")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(str(datasetname))
    plt.savefig("fig/"+ str(datasetname) +".png")
    plt.close()

def run():
    datasets = ["dataset/fecho1.txt", "dataset/fecho2.txt"]

    for dataset in datasets:
        # Dataset path
        INPUT_PATH = dataset
        # Load dataset
        points = np.loadtxt(INPUT_PATH).astype(np.float)
        # Call function to construct solution
        convex_hull_indexes = convex_hull(points)
        # Name of solution file
        output_file_name = 'fecho{}.txt'.format(INPUT_PATH[-5])
        # Function to save solution file
        print_solution("solution/{}".format(output_file_name), convex_hull_indexes)
        # Dataset name
        datasetname = output_file_name.split(".")[0]
        # Draw solution
        draw_polygon(points, convex_hull_indexes, datasetname)

run()
