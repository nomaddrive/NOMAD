import torch
import numpy as np
import sys
sys.path.append("/workspace") # User added this, keeping it just in case, though PYTHONPATH=. usually suffices
from baselines.eval.benchmark.metrics import compute_map_features

def test_mirror_fix():
    print("Testing Mirror Fix logic...")
    
    # 1. Original Boston-like Road (Right Hand Traffic)
    # Road edge is to our RIGHT.
    # Driving North (Up): Velocity (0, 1).
    # Right Curb: x=2. Line from (2, 0) to (2, 10).
    # Cross Product Check: (0, 10). Agent at (0, 5).
    # (0-2)*(10-0) - (5-0)*(2-2) = (-2)*10 - 0 = -20.
    # Negative = Left of line = On Road. (Correct).
    
    road_edge_polylines_orig = {
        "x": np.array([2.0, 2.0]),
        "y": np.array([0.0, 10.0]),
        "lengths": np.array([2]),
        "scenario_id": np.array([0])
    }
    
    # Agent at (0, 5) -> "Left" of the curb at x=2.
    x = np.array([[[0.0]]]) 
    y = np.array([[[5.0]]])
    heading = np.array([[[np.pi/2]]])
    scenario_ids = np.array([[0]])
    agent_length = np.array([4.0])
    agent_width = np.array([2.0])
    valid = np.array([[[True]]])
    
    device = torch.device('cpu')
    
    print("\n--- Case 1: Original (RHT) ---")
    dist_orig, offroad_orig = compute_map_features(
        x, y, heading, scenario_ids, agent_length, agent_width,
        road_edge_polylines_orig, device, valid
    )
    print(f"Distance: {dist_orig[0,0,0]} (Expected < 0)")
    print(f"Offroad: {offroad_orig[0,0,0]}")


    # 2. Mirrored Singapore (Simulating C++ Fix)
    # Mirror X: x -> -x.
    # Agent: (0, 5) -> (0, 5).
    # Curb: (2, 0)->(2, 10) becomes (-2, 0)->(-2, 10).
    #
    # WITHOUT C++ Fix (Point Reversal), vector is (-2,0) to (-2,10) (UP).
    # Agent (0,5) is to the RIGHT of x=-2. 
    # CP: (0 - -2)*(10-0) ... = (2)*10 = +20. Positive = Offroad. (THE BUG)
    #
    # WITH C++ Fix: Points are REVERSED.
    # Line is (-2, 10) to (-2, 0) (DOWN).
    # CP: (0 - -2)*(0-10) ... = (2)*(-10) = -20. Negative = On Road. (THE FIX)
    
    print("\n--- Case 2: Mirrored with Point Reversal (Simulating C++ Fix) ---")
    
    # Manually mirror AND reverse order
    road_edge_polylines_mirrored = {
        "x": np.array([-2.0, -2.0]),       # x -> -x (from 2, 2 to -2, -2)
        "y": np.array([10.0, 0.0]),        # Reversed Y order! (10 to 0)
        "lengths": np.array([2]),
        "scenario_id": np.array([0])
    }
    
    # Mirror agent (0 is 0)
    
    dist_mirror, offroad_mirror = compute_map_features(
        -x, y, heading, scenario_ids, agent_length, agent_width,
        road_edge_polylines_mirrored, device, valid
    )
    
    print(f"Distance: {dist_mirror[0,0,0]} (Expected < 0)")
    print(f"Offroad: {offroad_mirror[0,0,0]}")
    
    if dist_mirror[0,0,0] < 0:
        print("\n[SUCCESS] Reversing the points corrects the sign flip!")
    else:
        print("\n[FAILED] Still positive?")

if __name__ == "__main__":
    test_mirror_fix()
