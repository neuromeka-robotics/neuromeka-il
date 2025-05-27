POLICY = {
    "test_task": {
        "model": {
            "type": "act",
            "directory": "weights/test_task/2025-05-28-05-59-47",
            "weight": "policy_last.ckpt",
            "datastats": "dataset_stats.pkl",
            "configuration": "config.yaml",
        },
        "deploy": {
            "camera_serial": "233522079515",
            "success_threshold": 1.,
        },
    },
}
