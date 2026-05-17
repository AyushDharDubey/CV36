def get_focal_length(measured_distance, real_width, width_in_image):
    focal_length = (width_in_image * measured_distance) / real_width
    return focal_length


def get_distance(focal_Length, real_width, face_width_in_frame):
    distance = (real_width * focal_Length) / face_width_in_frame
    return distance


def speed(distance, takenTime):
    speed = distance/takenTime
    return speed