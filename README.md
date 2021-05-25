# lacing_vision
 Vision code for Robotic Lacing project
 
 To reference in GH via compas.RPC Proxy object:
 
 1) Install to your environment using:

(yourCondaEnv) ~YourGitHubRepo/lacing_vision/src>pip install -e .
(yourCondaEnv) >python -m compas_rhino.install -p lacing_vision

2) Create a GH python component with code such as:

        import compas
        import compas_rhino
        import rhinoscriptsyntax as rs

        from compas_rhino.geometry import RhinoPlane
        from abb_communication.clients.rfl_robot.communication.messages.messagetypes import *
        from compas.rpc import Proxy

        points = []

        if start:
            with Proxy('lacing_vision') as camera:
                planePt, planeNorm, pts = camera.find_corners()
            
                print planePt
                print planeNorm
                print pts

                for pt in pts:
                    points.append(rs.CreatePoint(pt))

                plane = rs.PlaneFromNormal(planePt, planeNorm)
                camera_compas_frame = RhinoPlane.from_geometry(plane).to_compas()
