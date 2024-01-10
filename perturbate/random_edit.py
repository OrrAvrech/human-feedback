import matplotlib.pyplot as plt
import torch
import joblib
from pathlib import Path
from models.smpl import SMPL, SMPLConfig
from dataclasses import asdict
from utils.visualize import viz_smplx, Renderer
from smplx.lbs import batch_rigid_transform
import numpy as np
from copy import deepcopy
# from phalp.configs.base import FullConfig
# from phalp.visualize.visualizer import Visualizer


class SMPLWrapper:
    def __init__(self, cfg: SMPLConfig):
        self.smpl = SMPL(**asdict(cfg))


def get_smpl_output(smpl_dir: Path, predictions: list[dict]):
    smpl_cfg = SMPLConfig(
        gender="neutral",
        joint_regressor_extra=smpl_dir / "SMPL_to_J19.pkl",
        # joint_regressor_extra=None,
        mean_params=smpl_dir / "smpl_mean_params.npz",
        model_path=smpl_dir / "smpl",
        num_body_joints=23,
    )

    global_orient = torch.Tensor(
        np.array(list(map(lambda x: x["global_orient"], predictions)))
    )
    body_pose = torch.Tensor(np.array(list(map(lambda x: x["body_pose"], predictions))))
    betas = torch.Tensor(np.array(list(map(lambda x: x["betas"], predictions))))

    pred_smpl_params = dict()
    pred_smpl_params["global_orient"] = global_orient
    pred_smpl_params["body_pose"] = body_pose
    pred_smpl_params["betas"] = betas

    smpl_model = SMPLWrapper(smpl_cfg)
    smpl_output = smpl_model.smpl(**pred_smpl_params, pose2rot=False)
    return smpl_output, smpl_model


def main():
    pkl_path = Path(
        "../4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    )
    smpl_dir = Path("./models/data")

    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())[0]

    pred_smpl_params = predictions["smpl"][0]

    # R_knee 5
    cp = deepcopy(pred_smpl_params)
    joint_idx = 15
    rknee = cp["body_pose"][joint_idx, :, :]
    # rankle = cp["body_pose"][:, 8, :, :]
    noise_magnitude = 0.5  # Adjust the magnitude of the noise
    direction = np.array([0.0, 0.0, 1.0])
    noisy_knee_rotation = rknee + noise_magnitude * direction
    pred_smpl_params["body_pose"][joint_idx, :, :] = noisy_knee_rotation

    smpl_output, smpl_model = get_smpl_output(smpl_dir, [pred_smpl_params])

    # img = viz_smplx(smpl_output, model=smpl_model.smpl)
    # viz_motion(smpl_output, smpl_model)
    #
    renderer = Renderer(focal_length=5000, img_res=256, faces=smpl_model.smpl.faces)
    vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()
    pred_cam_t = torch.Tensor([[[-1.8067e-02,  2.4293e-01,  2.7637e+01]]]).detach().cpu().numpy().squeeze()
    img = renderer(vertices=vertices, camera_translation=pred_cam_t)
    plt.imshow(img)
    plt.show()

    # cfg = FullConfig()
    # viz = Visualizer(cfg, smpl_model)
    # a, b = viz.render_video(predictions)
    # print(a)



if __name__ == "__main__":
    main()
