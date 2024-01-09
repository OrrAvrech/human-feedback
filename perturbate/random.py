import torch
import joblib
from pathlib import Path
from models.smpl import SMPL, SMPLConfig
from dataclasses import asdict
from utils.visualize import viz_smplx


def main():
    pkl_path = Path(
        "/Users/orrav/Documents/projects/4D-Humans/demo_10_most_common_YOGA_MISTAKES_new_students_make_and_how_to_fix_them_124.pkl"
    )
    smpl_dir = Path("/Users/orrav/Documents/projects/human-feedback/models/data")
    save_dir = Path("/Users/orrav/Documents/Data/human-feedback/experiments")

    data4d = joblib.load(pkl_path)
    pred = list(data4d.values())[0]

    smpl_cfg = SMPLConfig(gender="neutral",
                          joint_regressor_extra=smpl_dir / "SMPL_to_J19.pkl",
                          mean_params=smpl_dir / "smpl_mean_params.npz",
                          model_path=smpl_dir / "smpl",
                          num_body_joints=23)
    pred_smpl_params = pred["smpl"][0]

    pred_smpl_params["global_orient"] = torch.Tensor(pred_smpl_params["global_orient"]).reshape(
        1, -1, 3, 3
    )
    pred_smpl_params["body_pose"] = torch.Tensor(pred_smpl_params["body_pose"]).reshape(
        1, -1, 3, 3
    )
    pred_smpl_params["betas"] = torch.Tensor(pred_smpl_params["betas"]).reshape(1, -1)

    smpl_model = SMPL(**asdict(smpl_cfg))
    smpl_output = smpl_model(**pred_smpl_params, pose2rot=False)
    viz_smplx(smpl_output, model=smpl_model)


if __name__ == "__main__":
    main()
