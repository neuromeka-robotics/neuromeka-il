POLICY = {
    "test_task": {
        "model": {
            "type": "act",
            "directory": "weights/test_task/2025-02-25-18-33-59",
            "weight": "policy_dagger_1.ckpt",
            "datastats": "dataset_stats.pkl",
            "configuration": "config.yaml",
        },
        "deploy": {
            "camera_serial": "233522079515",
            "no_init_pose": False,
            "z_clipping": 313.61996,  # task space Z axis clipping [mm]
            "success_threshold": 0.05,
        },
    },
}
