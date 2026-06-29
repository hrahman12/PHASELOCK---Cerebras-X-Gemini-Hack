import math
import numpy as np

R_MAJOR = 3.0
R_MINOR = 1.0

def mesh_to_torus(mesh_x, mesh_y, W=125, H=120):
    """
    Topology-preserving WSE mesh to torus overlay mapping.
    Maps a 2D mesh coordinate to 3D torus Cartesian coordinates.
    """
    u = mesh_x / W
    v = mesh_y / H

    phi = 2 * math.pi * u
    theta = 2 * math.pi * v

    X = (R_MAJOR + R_MINOR * math.cos(theta)) * math.cos(phi)
    Y = (R_MAJOR + R_MINOR * math.cos(theta)) * math.sin(phi)
    Z = R_MINOR * math.sin(theta)

    return {
        "mesh_x": mesh_x,
        "mesh_y": mesh_y,
        "u": u,
        "v": v,
        "phi": phi,
        "theta": theta,
        "X": X,
        "Y": Y,
        "Z": Z,
    }

def calculate_mesh_distance(x1, y1, x2, y2, W=125, H=120):
    """
    Calculates Manhattan distance on the 2D wrapped torus mesh.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dx_wrapped = min(dx, W - dx)
    dy_wrapped = min(dy, H - dy)
    return dx_wrapped + dy_wrapped

def calculate_torus_distance(pos1, pos2):
    """
    Calculates Euclidean distance in 3D toroidal space.
    pos = (X, Y, Z)
    """
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)

def compute_distortion_metrics(pos_list, sample_size=200, W=125, H=120):
    """
    Computes global embedding distortion metrics:
    - mapping_loss: discrepancy between mesh and torus normalized distances.
    - neighbor_preservation_score: fraction of mesh-neighbors that remain 3D-neighbors.
    """
    # Max metrics for normalization
    max_mesh_dist = (W + H) / 2.0
    max_torus_dist = 2.0 * (R_MAJOR + R_MINOR)
    
    # 1. Estimate mapping loss via sampling pairs of agents
    losses = []
    for _ in range(sample_size):
        idx1 = random_idx = np.random.randint(0, len(pos_list))
        idx2 = random_idx = np.random.randint(0, len(pos_list))
        if idx1 == idx2:
            continue
        p1 = pos_list[idx1]
        p2 = pos_list[idx2]
        
        dm = calculate_mesh_distance(p1["mesh_x"], p1["mesh_y"], p2["mesh_x"], p2["mesh_y"], W, H)
        dt = calculate_torus_distance((p1["X"], p1["Y"], p1["Z"]), (p2["X"], p2["Y"], p2["Z"]))
        
        norm_dm = dm / max_mesh_dist
        norm_dt = dt / max_torus_dist
        losses.append(abs(norm_dm - norm_dt))
        
    mapping_loss = float(np.mean(losses)) if losses else 0.0
    
    # 2. Check neighbor preservation
    # For each node, its 4 grid neighbors are checked.
    # On our torus mapped grid, neighbor distance in 3D should be small.
    # Ideal grid step distance in 3D for W=125, H=120 is small (approx R_minor * 2pi/H = 0.05).
    # We define a threshold for neighbor preservation, e.g. 3D distance < 0.3.
    preservation_checks = 0
    preserved_count = 0
    
    for _ in range(sample_size):
        idx = np.random.randint(0, len(pos_list))
        p = pos_list[idx]
        px, py = p["mesh_x"], p["mesh_y"]
        pos_3d = (p["X"], p["Y"], p["Z"])
        
        # Define 4 physical neighbors on grid
        neighbors = [
            ((px + 1) % W, py),
            ((px - 1) % W, py),
            (px, (py + 1) % H),
            (px, (py - 1) % H)
        ]
        
        # Find these neighbors in our list (by grid coord lookup)
        # To make it fast, we rebuild a map if needed, but since it's just a test,
        # we can calculate their positions analytically.
        for nx, ny in neighbors:
            n_pos = mesh_to_torus(nx, ny, W, H)
            n_pos_3d = (n_pos["X"], n_pos["Y"], n_pos["Z"])
            dt = calculate_torus_distance(pos_3d, n_pos_3d)
            
            preservation_checks += 1
            if dt < 0.25: # Preserved neighbor threshold
                preserved_count += 1
                
    neighbor_score = (preserved_count / preservation_checks) if preservation_checks > 0 else 1.0
    
    return {
        "mapping_loss": round(mapping_loss, 4),
        "neighbor_preservation_score": round(neighbor_score, 4)
    }
