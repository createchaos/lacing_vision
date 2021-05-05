import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
#import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from skspatial.objects import Plane, Points

_pipe = None
_pipe = rs.pipeline()

def get_pipe():
    global _pipe
    return _pipe

# =============================================================

def find_corners():
    # Configure depth and color streams
    pipe = get_pipe()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipe.start(config)

    # Skip the first few frames to give the Auto-Exposure time to adjust
    for x in range(20):
        pipe.wait_for_frames()

    windowClose = False
    corners = []
    try:
        while windowClose == False:

            # Store next frameset for later processing
            frameset = pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            ### Do corner detection on unaligned image first ###
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 2)

            # Loop for detected corner points
            for i in corners:
                x, y = i.ravel()
                depth = depth_frame.get_distance(x,y)

                # Draw corner points in color image
                cv2.circle(color_image, (x,y), 3, (255,0,0), -1)
                color_point = np.asanyarray(i)

            # Show image
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(0)

            # Exit loop if window is closed
            if cv2.getWindowProperty('RealSense', 0) == -1:
                windowClose = True
    
    finally:

        # Cleanup
        pipe.stop()
        print("Frames Captured")

    finalFrameset = frameset
    print(corners)

    # Get depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
    color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

    # Align the streams
    align = rs.align(rs.stream.color)
    frameset = align.process(finalFrameset)

    aligned_depth_frame = frameset.get_depth_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    aligned_color_frame = frameset.get_color_frame()
    color_image = np.asanyarray(aligned_color_frame.get_data())

    # Depth data
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    
    list_x = []
    list_y = []
    list_z = []
    list_pcd = []
    for i in corners:
        x, y = i.ravel()
        depth = depth_frame.get_distance(x,y)

        # Project color pixel to depth image and return depth image pixel
        depth_px = rs.rs2_project_color_pixel_to_depth_pixel(
            aligned_depth_frame.get_data(),
            depth_scale,
            depth_min=0.0,
            depth_max=5.0,
            depth_intrin=depth_intrin,
            color_intrin=color_intrin,
            depth_to_color=depth_to_color_extrin,
            color_to_depth=color_to_depth_extrin,
            from_pixel=[x,y]
        )

        #xd = round(depth_px[0])
        #yd = round(depth_px[1])

        xd = int(x)
        yd = int(y)

        # Some depth image pixels are out of range, run only for postive, non-zero values
        if xd > 0:
            ddist = depth_frame.get_distance(xd, yd)
            print("at ", xd, ",", yd, " depth is ", ddist)

            if ddist > 0 and ddist < 1:
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xd, yd], ddist)

                print(depth_point)

                # Add coordinates to list
                list_x.append(depth_point[0])
                list_y.append(depth_point[1])
                list_z.append(depth_point[2])

                # Create list of points (future use for point cloud processing?)
                list_pcd.append(depth_point)

    
    # Configure 3d plot window
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Convert m to mm
    for pt in list_pcd:
        pt[0] = 1000*pt[0]
        pt[1] = 1000*pt[1]
        pt[2] = 1000*pt[2]


    # Plot Points
    ax.scatter(1000*list_x, 1000*list_y, 1000*list_z, c='red')
    plt.show() 

    # Find best fit plane through points
    bestPlane = Plane.best_fit(list_pcd)
    planePt = bestPlane.point
    planeVctr = bestPlane.normal

    planePt = [planePt[0], planePt[1], planePt[2]]
    planeVctr = [planeVctr[0], planeVctr[1], planeVctr[2]]

    # Format output data
    output = [planePt, planeVctr, list_pcd]
    print(output)

    # Output formatted as list of points [[planePt], [planeVctr], [[pt1], [pt2], [pt3], [pt4]]] in mm
    return output

#find_corners()