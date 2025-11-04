from fairino import Robot
import time

URL_ENV = '192.168.57.2'
# A connection is established with the robot controller. A successful connection returns a robot object
robot = Robot.RPC(URL_ENV)

# 