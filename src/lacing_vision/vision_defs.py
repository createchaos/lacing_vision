import cv2
import numpy as np
#import matplotlib.pyplot as plt
import pyrealsense2 as rs
#from mpl_toolkits.mplot3d import Axes3D
from skspatial.objects import Plane, Points

_pipe = None
_pipe = rs.pipeline()

# init values for selection mask
drawing = False
ix = 0
iy = 0
jx = 640
jy = 480

#__all__ = [drawing,ix,iy,jx,jy,get_pipe,config_camera,list_cameras,draw_rectangle_with_drag,get_corners_selmask,draw_corners,avg_nearestValidPts,corners_to_points,fitPlane]

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

def avg_nearestValidPts(frameset, x, y, maxDist=0.3, minDist=0.0, planeTh=0.01):
    pts = [x-1,y, x+1,y, x,y-1, x,y+1]
    nearPtDepth = []
    for i in range(4):
        depth = frameset.depth_frame.get_distance(pts[2*i], pts[2*i+1])
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

def corners_to_points(camera, frameset, corners, maxDist=0.3, minDist=0.0, planeTh=0.1):
    list_pts = []
    for i in corners:
        x,y = i.ravel()
        xd = int(x)
        yd = int(y)

        depth = frameset.depth_frame.get_distance(xd,yd)

        if depth < minDist or depth > maxDist:
            depth = avg_nearestValidPts(frameset, xd, yd, maxDist, minDist, planeTh)

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

#def find_corners(num_corners, maxDist, minDist):

#print(list_cameras())

#find_corners_selmask('048122071136')