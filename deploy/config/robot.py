ROBOT_ID = [0]
ROBOT_HOME_POS = {
    "test_task": {
        0: [-5.0032735, -20.997824, -84.92132, -0.020232247, -73.95311, -3.4574845],
    }
}

CONTROL = {
    "period": 0.05,
    "vel_scale": 1.,  # 0 ~ 1
    "acc_scale": 10.,  # 0 ~ 10
    "move_vel_scale": 50,  # 0 ~ 100
    "move_acc_scale": 50  # 0 ~ 1000
}

ROBOT_CONFIG = {
    0: {
        "ip": "192.168.0.135",
    },
}
