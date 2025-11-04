
from fairino import Robot
from flask import Flask, request, jsonify

ROBOT_URL = '192.168.57.2'
robot = Robot(ROBOT_URL)

## 使用flask 创建一个简单的机器人控制器API 初始化robot 