import cv2
import numpy as np
import pyrealsense2 as rs
from skspatial.objects import Plane, Points

import itertools

_pipe = None
_pipe = rs.pipeline()

# init values for selection mask
drawing = False
ix = 0
iy = 0
jx = 640
jy = 480

def dynamic_dist(devID, maxDist=0.3, minDist=0, width=640, height=480, fps=30):
    # init values for selection mask
    global jx, jy
    jx = width
    jy = height

    # init camera
    pipe = get_pipe()
    config = config_camera(devID, width, height, fps)
    profile = pipe.start(config)

    camera = Camera(profile)

    corners = []
    windowClose = False

    try:
        while windowClose == False:
            maskLoop = True
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('RealSense', draw_rectangle_with_drag)
            #global color_image
            global bothImgs

            while maskLoop == True:
                fs = Frameset(pipe)
                fs.align_streams()

                get_masked_frameset(fs)
                # corners = get_corners_selmask(fs)
                #draw_corners(fs, corners)

                bothImgs = np.hstack((fs.color_image, fs.depth_image))
                cv2.imshow('RealSense', bothImgs)
                k = cv2.waitKey(10)
                if cv2.getWindowProperty('RealSense', 0) == -1:
                    maskLoop = False
                if k == 32: # k == 32 is spacebar?
                    maskLoop = False
                    cv2.destroyWindow('RealSense')
                else:
                    maskLoop = True
            
            dimage = fs.depth_image
            masked = dimage[iy:jy, ix:jx]
            cv2.namedWindow('masked', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('masked', masked)
            # masked_depths = []
            # for i in range(iy, jy):
            #     for j in range(ix, jx):
            #         masked_depths[i,j] = fs.depth_frame.get_distance(j+ix,i+iy)
            # filter = depth_filter(masked_depths, maxDist, minDist)
            # masked_filtered = masked[filter]
            # #points = corners_to_points(camera, fs, masked_filtered, 1, 0.1)

            # for i in range(iy, jy):
            #     for j in range(ix, jx):
            #         depth_point = rs.rs2_deproject_pixel_to_point(camera.dintrin, [j+ix,i+iy], depth)
            # list_pts.append(depth_point)

            # list_ptcld = corners_to_points(camera, fs, corners, 1, 0.1)
            # draw_corners(fs, corners, 0.85, 0.1, text=True)

            # cv2.namedWindow('ConfirmImage', cv2.WINDOW_AUTOSIZE)
            # final = np.hstack((fs.color_image, fs.depth_image))
            # cv2.imshow('ConfirmImage', final)
            # if cv2.waitKey(0) == 32: # k == 27 is enter? 13 is enter
            #     windowClose = True
            #     cv2.destroyWindow('ConfirmImage')
            # else:
            #     windowClose = False
            
    finally:
        pipe.stop()

    #output = fitPlane(list_ptcld)

    #return output

# =============================================================

def depth_filter(arr, maxDist=0.3, minDist=0):
    filter_arr = []

    for element in arr:
        if element > maxDist or element < minDist:
            filter_arr.append(False)
        else:
            filter_arr.append(True)

    return filter_arr

# =============================================================

def find_corners(devID, num_corners, maxDist=0.3, minDist=0.1, width=640, height=480, fps=30):
    pipe = get_pipe()
    config = config_camera(devID, width, height, fps)
    profile = pipe.start(config)

    camera = Camera(profile)

    fs = Frameset(pipe)
    fs.align_streams()

    planeTh, maxPtDist = geom_plane(stripW=0.01, stripWTol=0.005, angleMax=90, angleMin=45, tiltAngle=45)
    corners = get_corners(fs, num_corners, width, height, quality=0.1, ptDistMin=2)
    corners = filter_corners(fs, corners, maxDist, minDist, planeTh)
    list_pts = corners_to_points(camera, fs, corners)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    draw_corners(fs, corners)

    format_for_opencv(fs)
    bothImgs = np.hstack((fs.color_image, fs.depth_image))
    cv2.imshow('RealSense', bothImgs)
    k = cv2.waitKey(0)

    print(list_pts)

    return list_pts

    """
    if len(corners) > 2:
        list_pts = corners_to_points(camera, fs, corners)
        if max(clusterLengths(list_pts)) < maxPtDist:
            validCluster = True
            return list_pts
        else:
            validCluster = False
            return None
    """

# =============================================================

def vecLen(pt1, pt2):
    length = np.sqrt(np.square(abs(pt1[0]-pt2[0]))+np.square(abs(pt1[1]-pt2[1]))+np.square(abs(pt1[2]-pt2[2])))

    return length

# =============================================================

def clusterLengths(points):
    lengths = []
    if len(points) > 2:
        # Find max and min distances between pts
        combs = list(itertools.combinations(points, 2))
        for cmb in combs:
            lengths.append(vecLen(cmb[0], cmb[1]))
        #print(lengths)

    elif len(points) == 2:
        lengths.append(vecLen(points[0], points[1]))

    return lengths
    
#points = [[0,0,0], [1,1,1], [1,0,0]]
#print(list(itertools.combinations(points, 2)))
#print(clusterLengths(points))

# =============================================================

def robot_vision(devID1,devID2):
    ctx = rs.context()

    config = config_camera(devID, width, height, fps)
    pipe = get_pipe()
    profile = pipe.start(config)

    camera = Camera(profile)

# =============================================================

def find_corners_selmask(devID, width=640, height=480, fps=30, searchRad=5):
    # init values for selection mask
    global jx, jy
    jx = width
    jy = height

    # init camera
    pipe = get_pipe()
    config = config_camera(devID, width, height, fps)
    profile = pipe.start(config)

    camera = Camera(profile)

    corners = []
    windowClose = False

    try:
        while windowClose == False:
            maskLoop = True
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('RealSense', draw_rectangle_with_drag)
            #global color_image
            global bothImgs

            while maskLoop == True:
                fs = Frameset(pipe)
                fs.align_streams()

                corners = get_corners_selmask(fs)
                draw_corners(fs, corners, 1, 0)

                bothImgs = np.hstack((fs.color_image, fs.depth_image))
                cv2.imshow('RealSense', bothImgs)
                k = cv2.waitKey(10)
                if cv2.getWindowProperty('RealSense', 0) == -1:
                    maskLoop = False
                if k == 32: # k == 32 is spacebar?
                    maskLoop = False
                    cv2.destroyWindow('RealSense')
                else:
                    maskLoop = True

            list_ptcld = corners_to_points(camera, fs, corners, 1, 0, searchRad)
            print(list_ptcld)
            print(corners)
            draw_corners(fs, corners, 1, 0, text=True)

            cv2.namedWindow('ConfirmImage', cv2.WINDOW_AUTOSIZE)
            final = np.hstack((fs.color_image, fs.depth_image))
            cv2.imshow('ConfirmImage', final)
            if cv2.waitKey(0) == 32: # k == 27 is enter? 13 is enter
                windowClose = True
                cv2.destroyWindow('ConfirmImage')
            else:
                windowClose = False
            
    finally:
        pipe.stop()

    output = fitPlane(list_ptcld)

    return output

# =============================================================

def get_pipe():
    global _pipe

    return _pipe

# =============================================================

def config_camera(devID, width=640, height=480, fps=30):
    # devID is a string, use list_cameras to get devIDs
    config = rs.config()
    config.enable_device(devID)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    return config


# =============================================================

class Camera:
    def __init__(self, profile, width=640, height=480):
        # profile must have a depth and a color stream
        self.dsensor = profile.get_device().first_depth_sensor()
        self.rgbsensor = profile.get_device().query_sensors()[1]
        self.rgbsensor.set_option(rs.option.enable_auto_exposure, False)
        self.rgbsensor.set_option(rs.option.exposure, 500)
        self.dscale = self.dsensor.get_depth_scale()

        self.dintrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.cintrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.dtoc_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
        self.ctod_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

        self.width = width
        self.height = height

# =============================================================

def list_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()

    devIDs = []
    for i in devices:
        serNum = i.get_info(rs.camera_info.serial_number)
        devIDs.append(serNum)

    return devIDs

# =============================================================

def draw_rectangle_with_drag(event, x, y, flags, param):
      
        global ix, iy, drawing, bothImgs, jx, jy#, passImage
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix = x
            iy = y      
            #passImage = False
            cv2.waitKey(0)    
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(bothImgs, (ix,iy), (x,iy), (255,0,0), 3)
                cv2.line(bothImgs, (ix,iy), (ix,y), (255,0,0), 3)
                cv2.imshow('RealSense', bothImgs)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(bothImgs, pt1 = (ix, iy), pt2 = (x, y), color = (0,0,255), thickness  = 2)
            cv2.rectangle(bothImgs, pt1 = (ix+640,iy), pt2 = (x+640, y), color =(0,0,255), thickness = 2)
            jx = x
            jy = y
            cv2.imshow('RealSense', bothImgs)
            cv2.waitKey(10)

# =============================================================

class Frameset:
    def __init__(self, pipe=_pipe):
        global _pipe

        self.frameset = pipe.wait_for_frames()
        self.color_frame = self.frameset.get_color_frame()
        self.depth_frame = self.frameset.get_depth_frame()

        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())

    def align_streams(self):
        align = rs.align(rs.stream.color)
        self.frameset = align.process(self.frameset)

        self.depth_frame = self.frameset.get_depth_frame()
        self.color_frame = self.frameset.get_color_frame()

        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())

    def set_colImg(self, image):
        self.color_image = image

    def set_depImg(self, image):
        self.depth_image = image

# =============================================================

def get_corners_selmask(frameset, num_corners=4, width=640, height=480, quality=0.01, ptDistMin=2):
    color_image = frameset.color_image
    roi = color_image[iy:jy, ix:jx]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[iy:jy, ix:jx] = 255
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, num_corners, quality, ptDistMin, mask=mask)
    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_image[iy:jy, ix:jx] = roi
    frameset.set_colImg(color_image)

    colorizer = rs.colorizer()
    depth_image = np.asanyarray(colorizer.colorize(frameset.depth_frame).get_data())
    roi_depth = depth_image[iy:jy, ix:jx]
    gray_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    depth_image = cv2.cvtColor(gray_depth, cv2.COLOR_GRAY2BGR)
    depth_image[iy:jy, ix:jx] = roi_depth
    frameset.set_depImg(depth_image)

    return corners

# =============================================================

def get_edges(frameset, threshold1=1, threshold2=3, width=640, height=480, aperatureSize=3):
    color_image = frameset.color_image
    roi = color_image[iy:jy, ix:jx]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[iy:jy, ix:jx] = 255
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blurred)
    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_image[iy:jy, ix:jx] = roi
    frameset.set_colImg(color_image)

    colorizer = rs.colorizer()
    depth_image = np.asanyarray(colorizer.colorize(frameset.depth_frame).get_data())
    roi_depth = depth_image[iy:jy, ix:jx]
    gray_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    depth_image = cv2.cvtColor(gray_depth, cv2.COLOR_GRAY2BGR)
    depth_image[iy:jy, ix:jx] = roi_depth
    frameset.set_depImg(depth_image)

    return edges

def auto_canny(image, sigma=0.33):
    #compute the median of the single channel pixel intensities
    v = np.median(image)

    #apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma)*v))
    upper = int(min(255, (1.0 + sigma)*v))
    edged = cv2.Canny(image, lower, upper)

    return edged

# =============================================================

def get_masked_frameset(frameset, width=640, height=480):
    color_image = frameset.color_image
    roi = color_image[iy:jy, ix:jx]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[iy:jy, ix:jx] = 255
    color_image[iy:jy, ix:jx] = roi
    frameset.set_colImg(color_image)

    colorizer = rs.colorizer()
    depth_image = np.asanyarray(colorizer.colorize(frameset.depth_frame).get_data())
    roi_depth = depth_image[iy:jy, ix:jx]
    gray_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    depth_image = cv2.cvtColor(gray_depth, cv2.COLOR_GRAY2BGR)
    depth_image[iy:jy, ix:jx] = roi_depth
    frameset.set_depImg(depth_image)

# =============================================================

def format_for_opencv(frameset):
    colorizer = rs.colorizer()
    depth_image = np.asanyarray(colorizer.colorize(frameset.depth_frame).get_data())
    frameset.set_depImg(depth_image)

# =============================================================

#planeTh, maxPtDist = geom_plane(stripW=0.01, stripWTol=0.005, angleMax=90, angleMin=45)
def geom_plane(stripW=0.01, stripWTol=0.005, angleMax=90, angleMin=45, tiltAngle=45):
    # max corner to corner length when angle is small, plane is straight-on
    planeL_1a = (stripW+stripWTol)/(np.cos(np.deg2rad(90)-np.deg2rad(angleMax)))
    planeL_1b = (stripW-stripWTol)/(np.cos(np.deg2rad(90)-np.deg2rad(angleMax)))

    planeL_2a = (stripW+stripWTol)/(np.cos(np.deg2rad(90)-np.deg2rad(angleMin)))
    planeL_2b = (stripW-stripWTol)/(np.cos(np.deg2rad(90)-np.deg2rad(angleMin)))

    maxPtToPt_1a = 2*(np.sin((np.deg2rad(180)-np.deg2rad(angleMax))/2))*planeL_1a
    maxPtToPt_1b = 2*(np.sin((np.deg2rad(180)-np.deg2rad(angleMax))/2))*planeL_1b

    maxPtToPt_2a = 2*(np.sin((np.deg2rad(180)-np.deg2rad(angleMin))/2))*planeL_2a
    maxPtToPt_2b = 2*(np.sin((np.deg2rad(180)-np.deg2rad(angleMin))/2))*planeL_2b

    maxPtToPt = max(maxPtToPt_1a, maxPtToPt_1b, maxPtToPt_2a, maxPtToPt_2b)

    """
    minPtToPt_1a = 2*(np.sin(np.deg2rad(angleMax)/2))*planeL_1a
    minPtToPt_1b = 2*(np.sin(np.deg2rad(angleMax)/2))*planeL_1b

    minPtToPt_2a = 2*(np.sin(np.deg2rad(angleMin)/2))*planeL_2a
    minPtToPt_2b = 2*(np.sin(np.deg2rad(angleMin)/2))*planeL_2b

    minPtToPt = min(minPtToPt_1a, minPtToPt_1b, minPtToPt_2a, minPtToPt_2b)
    """

    #If plane is tilted away at 45deg angle, find max depth diff
    maxDepthDiff = maxPtToPt*np.cos(np.deg2rad(tiltAngle))

    return maxDepthDiff, maxPtToPt

# =============================================================

def get_corners(frameset, num_corners=4, width=640, height=480, quality=0.01, ptDistMin=2):
    color_image = frameset.color_image
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, num_corners, quality, ptDistMin)

    return corners

# =============================================================

def filter_corners(frameset, corners, maxDist=0.3, minDist=0.0, planeTh=0.01):
    print(maxDist, minDist, planeTh)
    list_remove = []
    for i in range(len(corners)):
        x,y = corners[i].ravel()
        depth = frameset.depth_frame.get_distance(x,y)
        #print(depth)

        if depth < minDist or depth > maxDist:
            depth = avg_nearestValidPts(frameset, int(x), int(y), maxDist, minDist, planeTh)

        if depth < minDist or depth > maxDist:
            list_remove.append(i)
    
    #corners.delete(list_remove)
    np.delete(corners, list_remove)

    return corners

    


# =============================================================

def draw_corners(frameset, corners, depthMax=0.3, depthMin=0, text=False):
    if len(corners) > 0:
        xList = []
        for i in corners:
            x,y = i.ravel()
            xList.append(x)
        xAvg = sum(xList)/len(xList)
    else:
        xAvg = 0

    for i in corners:
        x, y = i.ravel()
        depth = frameset.depth_frame.get_distance(x,y)
        color_image = frameset.color_image
        depth_image = frameset.depth_image

        if depth < depthMin or depth > depthMax:
            cv2.circle(color_image, (x,y), 3, (0,0,255), -1)
            cv2.circle(depth_image, (x,y), 3, (0,0,255), -1)
        else:
            cv2.circle(color_image, (x,y), 3, (255,0,0), -1)
            cv2.circle(depth_image, (x,y), 3, (255,0,0), -1)

        if text:
            depthText = "{:.1f}".format(1000*depth)
            if x > xAvg:
                cv2.putText(depth_image, depthText, (int(x+10), int(y+3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(depth_image, depthText, (int(x-45), int(y+3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1, cv2.LINE_AA)
        
        frameset.set_colImg(color_image)
        frameset.set_depImg(depth_image)
        
# =============================================================
# x = 200
# y = 150
# i = 2
# pts = []
# x_list = list(range(x-i, x+i+1))
# y_list = list(range(y-i, y+i+1))
# pts = list(itertools.product(x_list, y_list))
# print(pts)
# print(pts[0][0], pts[0][1])
# print(pts[1][0], pts[1][1])


def avg_nearestValidPts(frameset, x, y, searchRadius=3, maxDist=0.3, minDist=0.0, planeTh=0.1):
    sR = searchRadius
    x_list = list(range(x-sR, x+sR+1))
    y_list = list(range(y-sR, y+sR+1))
    pts = list(itertools.product(x_list, y_list))

    nearPtDepth = []
    for pt in pts:
        depth = frameset.depth_frame.get_distance(pt[0], pt[1])
        if depth > minDist and depth < maxDist:
            nearPtDepth.append(depth)
    
    # If more than 1 valid point nearby, assume min depth is actually strip/plane
    # Find points more than planeTh m deeper than min and throw away
    if len(nearPtDepth) > 1:
        diff = max(nearPtDepth) - min(nearPtDepth)
        if diff > planeTh:
            realVal = min(nearPtDepth)
            for i in range(len(nearPtDepth)):
                ptDiff = nearPtDepth[i] - realVal
                if ptDiff > planeTh:
                    nearPtDepth.remove(i)
    
    if len(nearPtDepth) > 0:
        avgNearPtsDepth = sum(nearPtDepth)/len(nearPtDepth)
    else:
        avgNearPtsDepth = 0.0

    return avgNearPtsDepth

# =============================================================

def corners_to_points(camera, frameset, corners, maxDist=1, minDist=0.0, searchRadius=3, planeTh=0.1):
    list_pts = []
    for i in corners:
        x,y = i.ravel()
        xd = int(x)
        yd = int(y)

        depth = frameset.depth_frame.get_distance(xd,yd)

        if depth < minDist or depth > maxDist:
            depth = avg_nearestValidPts(frameset, xd, yd, searchRadius, maxDist, minDist, planeTh)

        if depth > minDist and depth < maxDist:
            depth_point = rs.rs2_deproject_pixel_to_point(camera.dintrin, [xd,yd], depth)
            list_pts.append(depth_point)
    
    # Convert m to mm
    for pt in list_pts:
        pt[0] = 1000*pt[0]
        pt[1] = 1000*pt[1]
        pt[2] = 1000*pt[2]

    return list_pts
        
# =============================================================

def fitPlane(list_pts):
    # Find best fit plane through points
    bestPlane = Plane.best_fit(list_pts)
    planePt = bestPlane.point
    planeVctr = bestPlane.normal

    planePt = [planePt[0], planePt[1], planePt[2]]
    planeVctr = [planeVctr[0], planeVctr[1], planeVctr[2]]

    # Format output data
    output = [planePt, planeVctr, list_pts]
    print(output)

    # Output formatted as list of points [[planePt], [planeVctr], [[pt1], [pt2], [pt3], [pt4]]] in mm
    return output

# =============================================================

#geom_plane(angleMax=95, angleMin=60)

#print(list_cameras())

#find_corners_selmask('849312070057')
#find_corners_selmask('048122071136')
#find_corners_selmask('935322071366')

#find_corners('048122071136', 300)    

#dynamic_dist('935322071366')