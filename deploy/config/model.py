POLICY = {
    "test_policy": {
        "model": {
            "type": "act",
            "directory": "train/weights/test_policy/2025-02-25-18-33-59",
            "weight": "policy_dagger_1.ckpt",
            "datastats": "dataset_stats.pkl",
            "configuration": "config.yaml",
        },
        "deploy": {
            "camera_serial": "231522073060",
            "no_init_pose": False,
            "z_clipping": 313.61996,  # task space Z axis clipping [mm]
            "success_threshold": 0.05,
        },
    },
}
