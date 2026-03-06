from isaaclab.managers import SceneEntityCfg

base = "Base"
hip = "_Hip"
shank = "_Shank"
thigh = "_Thigh"
foot = "_Foot"

class AygParams:
    base_name = f"{base}"
    hip_name = f".*{hip}"
    shank_names = f".*{shank}"
    thigh_names = f".*{thigh}"
    feet_names = f".*{foot}"
    
    lf_foot_name = f"LF{foot}"
    rf_foot_name = f"RF{foot}"
    lh_foot_name = f"LH{foot}"
    rh_foot_name = f"RH{foot}"
    
    undesired_contact_names = [f".*{thigh}", f".*{shank}"]
    termination_contact_names = [f"{base}", f".*{hip}", f".*{thigh}", f".*{shank}"]
    height_scanner = SceneEntityCfg("height_scanner")
