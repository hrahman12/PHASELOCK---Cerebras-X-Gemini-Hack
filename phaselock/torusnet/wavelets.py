from torusnet.embedding import mesh_to_torus
from torusnet.colors import COLOR_WAVELET

class WaveletFactory:
    def __init__(self, W=125, H=120):
        self.W = W
        self.H = H

    def create_wavelet(self, wavelet_type, agent_id, mesh_x, mesh_y, radius=3, amplitude=1.0, decay=0.2, ttl=5):
        """
        Creates a local control wavelet emitted by the Fusion Core.
        """
        pos = mesh_to_torus(mesh_x, mesh_y, self.W, self.H)
        
        return {
            "type": wavelet_type,
            "origin_agent": agent_id,
            "origin_mesh": (mesh_x, mesh_y),
            "origin_torus": (pos["X"], pos["Y"], pos["Z"]),
            "radius": radius,
            "amplitude": float(amplitude),
            "decay": float(decay),
            "ttl": int(ttl),
            "affected_agents": [],
            "color": COLOR_WAVELET,
        }

    def apply_wavelet_effects(self, wavelet, agents_list, mesh):
        """
        Applies local feedback controls to agents inside the wavelet's influence radius.
        """
        wx, wy = wavelet["origin_mesh"]
        rad = wavelet["radius"]
        amp = wavelet["amplitude"]
        
        affected = []
        for agent in agents_list:
            ax, ay = agent.x, agent.y
            
            # Manhattan distance on wrapped grid
            dx = abs(ax - wx)
            dy = abs(ay - wy)
            dx_wrapped = min(dx, self.W - dx)
            dy_wrapped = min(dy, self.H - dy)
            dist = dx_wrapped + dy_wrapped
            
            if dist <= rad:
                affected.append(agent.agent_id)
                # Compute distance attenuation
                attenuation = max(0.0, amp - (dist * wavelet["decay"]))
                
                # Apply specific wavelet behaviors:
                if wavelet["type"] == "THROTTLE":
                    # Slow down agent, reduce local queue congestion
                    agent.confidence = max(0.1, agent.confidence - 0.05 * attenuation)
                    mesh.state_array[ay, ax] = 2 # Set to CONGESTED (throttled state)
                elif wavelet["type"] == "COOLING":
                    # Sleep low-value redundant agents
                    if agent.confidence < 0.5:
                        agent.state = "SLEEPING"
                        mesh.state_array[ay, ax] = 0 # SLEEPING
                elif wavelet["type"] == "AMPLIFY":
                    # Boost confidence of high-performing leaders
                    agent.confidence = min(1.0, agent.confidence + 0.08 * attenuation)
                    agent.state = "ACTIVE"
                    mesh.state_array[ay, ax] = 1 # ACTIVE
                elif wavelet["type"] == "VERIFY":
                    # Ask neighbors to run consensus checks
                    agent.state = "ACTIVE"
                    mesh.state_array[ay, ax] = 1
                    agent.confidence = min(0.95, agent.confidence + 0.03 * attenuation)
                elif wavelet["type"] == "REROUTE":
                    # Signal routing nodes to bypass local hotspot
                    mesh.queue_size_array[ay, ax] = max(0, mesh.queue_size_array[ay, ax] - 2)
                elif wavelet["type"] == "CONSENSUS":
                    # Signal regions to summarize details
                    agent.confidence = min(1.0, agent.confidence + 0.05 * attenuation)
                    
        wavelet["affected_agents"] = affected
        return affected
