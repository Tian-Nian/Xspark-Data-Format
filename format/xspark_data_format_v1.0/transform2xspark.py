import cv2
import h5py
import json
import numpy as np
import os

def X_spark_format_pipeline(data_dict, save_path, episode_id):
    left_eef, left_joint, left_gripper, = data_dict["left_arm"]["eef"], data_dict["left_arm"]["joint"], \
                                                        data_dict["left_arm"]["gripper"]
    right_eef, right_joint, right_gripper = data_dict["right_arm"]["eef"], data_dict["right_arm"]["joint"],\
                                                        data_dict["right_arm"]["gripper"]

    left_gripper = np.array(left_gripper).reshape(-1, 1) # [m,] -> [m, 1]
    right_gripper = np.array(right_gripper).reshape(-1, 1) # [m,] -> [m, 1]

    def encode(imgs):
        if isinstance(imgs, bytes) or (isinstance(imgs,np.ndarray) and len(imgs.shape) == 1):
            return imgs
        encode_data = []
        for i in range(len(imgs)):
            success, encoded_image = cv2.imencode('.jpg', imgs[i])
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
        return np.array(encode_data)
    
    cam_head_color = encode(data_dict["cam_head"]["color"])
    cam_left_wrist_color = encode(data_dict["cam_left_wrist"]["color"])
    cam_right_wrist_color = encode(data_dict["cam_right_wrist"]["color"])

    subtasks = data_dict["extra_episode_info"].get("subtasks", [])
    instructions = data_dict["extra_episode_info"].get("instructions", [])

    hdf5_path = os.path.join(save_path, f"episode_{episode_id:07d}.hdf5") # 变成episode_000000 + episode_id .hdf5
    '''
    vision:
        head:
            colors:
            depths:    
            intrinsic_matrix:
            extrinsics_matrix:
            shape:
        left_wrist:
        right_wrist:
        (单臂)wrist:
        (optional)third_view:
    state:
        left_arm_joint_states:
        left_ee_joint_states: # 末端执行器关节状态（比如手的关节角）
        left_ee_poses: # 世界坐标系pose（xyz,qw,qx,qy,qz）
        left_tcp_poses:
        left_delta_ee_poses:   
        right_arm_joint_states:
        right_ee_joint_states:
        right_ee_poses:
        right_tcp_poses:
        right_delta_ee_poses:
    '''

    def get_cam_shape(img_bytes):
        if isinstance(img_bytes, np.ndarray):
            return img_bytes.shape  
        else:
            jpeg_bytes = img_bytes.rstrip(b"\0")
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            shape = cv2.imdecode(nparr, 1).shape
            return shape

    with h5py.File(hdf5_path, "w") as f:
        vision = f.create_group("vision")
        state = f.create_group("state")
        cam_head = vision.create_group("cam_head")
        cam_head.create_dataset("colors", data=cam_head_color)
        
        cam_head.create_dataset("shape", data=get_cam_shape(cam_head_color[0]))

        cam_left_wrist = vision.create_group("cam_left_wrist")
        cam_left_wrist.create_dataset("colors", data=cam_left_wrist_color)
        cam_left_wrist.create_dataset("shape", data=get_cam_shape(cam_left_wrist_color[0])) # 固定分辨率

        cam_right_wrist = vision.create_group("cam_right_wrist")
        cam_right_wrist.create_dataset("colors", data=cam_right_wrist_color)
        cam_right_wrist.create_dataset("shape", data=get_cam_shape(cam_right_wrist_color[0])) # 固定分辨率
        
        state.create_dataset("left_arm_joint_states", data=left_joint)
        state.create_dataset("left_ee_joint_states", data=left_gripper)
        state.create_dataset("left_ee_poses", data=left_eef)
        state.create_dataset("right_arm_joint_states", data=right_joint)
        state.create_dataset("right_ee_joint_states", data=right_gripper)
        state.create_dataset("right_ee_poses", data=right_eef)

        f.create_dataset("instructions", data=np.string_(json.dumps(instructions)))
        f.create_dataset("subtasks", data=np.string_(json.dumps(subtasks)))
        addition_info = f.create_group("additional_info")
        addition_info.create_dataset("frequency", data=data_dict.extra_episode_info.get("additional_info", {}).get("frequency", 30))
        f.create_dataset("data_format_version", data=np.string_(data_dict.extra_episode_info.get("data_format_version", "v1.0")))