import cv2
import roslibpy
import os
import numpy as np
import base64
import json

client = roslibpy.Ros(host='localhost', port=9090)
client.run()

movepose_srv = roslibpy.Service(client, '/simulation_bridge/move_robot_pose', 'welding_sim_gz/MoveRobotPose')

waypoint_file = open("example_wp.json")
waypoints = json.load(waypoint_file)

for i in range(0, len(waypoints)):
    raw_image = movepose_srv.call(waypoints[i])

    #Decode Image Data
    decoded_data = base64.b64decode(raw_image['result']['data'])
    np_data = np.frombuffer(decoded_data,np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    filename = "image_" + str(i) + ".jpg"
    cv2.imwrite(os.path.join("captured_image", filename), img)
    print("Saving wp", str(i), "image")

client.terminate()
