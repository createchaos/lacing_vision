import cv2
import numpy as np
#%matplotlib inline
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

# Need to fix finding corners on unaligned frame
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

            # Search four closest points if ddist = 0
            if ddist == 0.0:
                pts = [xd-1,yd, xd+1,yd, xd,yd-1, xd,yd+1]
                nearDist = []
                for i in range(4):
                    dist = depth_frame.get_distance(pts[2*i], pts[2*i+1])
                    if dist > 0 and dist < 1:
                        nearDist.append(dist)
                
                if len(nearDist) > 0:
                    newDdist = sum(nearDist)/len(nearDist)
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xd, yd], newDdist)

                    # Add coordinates to list
                    list_x.append(depth_point[0])
                    list_y.append(depth_point[1])
                    list_z.append(depth_point[2])

                    # Create list of points (future use for point cloud processing?)
                    list_pcd.append(depth_point)

            if ddist > 0 and ddist < 1:
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xd, yd], ddist)

                print(depth_point)

                # Add coordinates to list
                list_x.append(depth_point[0])
                list_y.append(depth_point[1])
                list_z.append(depth_point[2])

                # Create list of points (future use for point cloud processing?)
                list_pcd.append(depth_point)

    # Convert m to mm
    for pt in list_pcd:
        pt[0] = 1000*pt[0]
        pt[1] = 1000*pt[1]
        pt[2] = 1000*pt[2]
   
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


drawing = False
ix = 0
iy = 0
jx = 640
jy = 480

def find_corners_selmask():
    # Configure depth and color streams
    pipe = get_pipe()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    depthFilter = 0.3

    # Start streaming
    profile = pipe.start(config)

    # Skip the first few frames to give the Auto-Exposure time to adjust
    for x in range(20):
        pipe.wait_for_frames()

    # Get depth sensor's depth scale, intrinsic, and extrinsic profiles
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
    color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

    # Define selection mask drawing function
    def draw_rectangle_with_drag(event, x, y, flags, param):
      
        global ix, iy, drawing, color_image, jx, jy, passImage
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix = x
            iy = y      
            passImage = False
            cv2.waitKey(0)    
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(both, (ix,iy), (x,iy), (255,0,0), 3)
                cv2.line(both, (ix,iy), (ix,y), (255,0,0), 3)
                cv2.imshow('RealSense', both)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(both, pt1 = (ix, iy), pt2 = (x, y), color = (0,0,255), thickness  = 2)
            cv2.rectangle(both, pt1 = (ix+640,iy), pt2 = (x+640, y), color =(0,0,255), thickness = 2)
            jx = x
            jy = y
            cv2.imshow('RealSense', both)
            cv2.waitKey(10)

    corners = []
    windowClose = False
    try:
        while windowClose == False:
            maskLoop = True
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('RealSense', draw_rectangle_with_drag)
            global color_image
            global both

            while maskLoop == True:
                #print("in mask loop")
                # Store next frameset for later processing
                frameset = pipe.wait_for_frames()
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()

                # Align the streams
                align = rs.align(rs.stream.color)
                frameset = align.process(frameset)

                aligned_depth_frame = frameset.get_depth_frame()
                depth_image = np.asanyarray(aligned_depth_frame.get_data())

                aligned_color_frame = frameset.get_color_frame()
                color_image = np.asanyarray(aligned_color_frame.get_data())

                ### Do corner detection first ###
                color_image = np.asanyarray(aligned_color_frame.get_data())
                roi = color_image[iy:jy, ix:jx]
                mask = np.zeros((480,640), dtype=np.uint8)
                mask[iy:jy, ix:jx] = 255
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 2, mask=mask)
                color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                color_image[iy:jy, ix:jx] = roi

                colorizer = rs.colorizer()
                depth_image = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
                roi_depth = depth_image[iy:jy, ix:jx]
                gray_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
                depth_image = cv2.cvtColor(gray_depth, cv2.COLOR_GRAY2BGR)
                depth_image[iy:jy, ix:jx] = roi_depth


                # Move detected corners back to full image location and draw
                for i in corners:
                    x, y = i.ravel()
                    depth = aligned_depth_frame.get_distance(x,y)

                    # Draw corner points in color image
                    if depth == 0 or depth > depthFilter:
                        cv2.circle(color_image, (x,y), 3, (0,0,255), -1)
                    else:
                        cv2.circle(color_image, (x,y), 3, (255,0,0), -1)
                        

                    # Draw corner points in depth image
                    if depth == 0 or depth > depthFilter:
                        cv2.circle(depth_image, (x,y), 3, (0,0,255), -1)
                    else:
                        cv2.circle(depth_image, (x,y), 3, (255,0,0), -1)
                    #color_point_depth 

                # Show image
                both = np.hstack((color_image, depth_image))
                cv2.imshow('RealSense', both)    
                #cv2.imshow('RealSense', color_image)               
                k = cv2.waitKey(10)

                # Exit loop if window is closed
                if cv2.getWindowProperty('RealSense', 0) == -1:
                    maskLoop = False

                # Exit loop if spacebar is pressed
                if k == 32:
                    maskLoop = False
                    cv2.destroyWindow('RealSense')

            list_x = []
            list_y = []
            list_z = []
            list_pcd = []
            cornersList = []
            for i in corners:
                x, y = i.ravel()
                depth = aligned_depth_frame.get_distance(x,y)

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

                xd = int(x)
                yd = int(y)
                cornersList.append([xd,yd])

                # Some depth image pixels are out of range, run only for postive, non-zero values
                if xd > 0:
                    ddist = aligned_depth_frame.get_distance(xd, yd)

                    pts = [xd-1,yd, xd+1,yd, xd,yd-1, xd,yd+1]
                    nearDist = []
                    for i in range(4):
                        dist = aligned_depth_frame.get_distance(pts[2*i], pts[2*i+1])
                        if dist > 0 and dist < 1:
                            nearDist.append(dist)

                    # Search four closest points if ddist = 0
                    if ddist == 0.0:
                        
                        if len(nearDist) > 0:

                            # Find points more than 10mm deeper than min and throw away
                            if len(nearDist) > 1:
                                diff = max(nearDist) - min(nearDist)
                                if diff > 0.1:
                                    realVal = min(nearDist)
                                    for i in range(len(nearDist)):
                                        ptDiff = nearDist[i] - min(nearDist)
                                        if ptDiff > 0.1:
                                            nearDist.remove(i)

                            newDdist = sum(nearDist)/len(nearDist)

                            if newDdist > 0 and newDdist < depthFilter:
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xd, yd], newDdist)

                                # Add coordinates to list
                                list_x.append(depth_point[0])
                                list_y.append(depth_point[1])
                                list_z.append(depth_point[2])

                                # Create list of points (future use for point cloud processing?)
                                list_pcd.append(depth_point)
                        

                    if ddist > 0 and ddist < depthFilter:
                        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xd, yd], ddist)

                        # Add coordinates to list
                        list_x.append(depth_point[0])
                        list_y.append(depth_point[1])
                        list_z.append(depth_point[2])

                        # Create list of points (future use for point cloud processing?)
                        list_pcd.append(depth_point)
            # Convert m to mm
            for pt in list_pcd:
                pt[0] = 1000*pt[0]
                pt[1] = 1000*pt[1]
                pt[2] = 1000*pt[2]

            #print(cornersList)
            if len(cornersList) > 0:
                xAvg = sum(i[0] for i in cornersList)/len(cornersList)
            else:
                xAvg = 0

            for i in corners:
                x, y = i.ravel()
                depth = aligned_depth_frame.get_distance(x,y)

                # Draw corner points in color image
                if depth > depthFilter:
                    cv2.circle(color_image, (x,y), 3, (0,0,255), -1)
                else:
                    cv2.circle(color_image, (x,y), 3, (255,0,0), -1)


                # Draw corner points in depth image
                if depth > depthFilter:
                    cv2.circle(depth_image, (x,y), 3, (0,0,255), -1)
                else:
                    cv2.circle(depth_image, (x,y), 3, (255,0,0), -1)

                depthText = "{:.1f}".format(1000*depth)
                if x > xAvg:
                    cv2.putText(depth_image, depthText, (int((x+10)),int(y+3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(depth_image, depthText, (int((x-45)),int(y+3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1, cv2.LINE_AA)


            cv2.namedWindow('ConfirmImage', cv2.WINDOW_AUTOSIZE)
            final = np.hstack((color_image, depth_image))
            cv2.imshow('ConfirmImage', final)
            if cv2.waitKey(0) == 27:
                windowClose = False
                cv2.destroyWindow('ConfirmImage')
            else:
                windowClose = True
    
    finally:

        # Cleanup
        pipe.stop()

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

#find_corners_selmask()