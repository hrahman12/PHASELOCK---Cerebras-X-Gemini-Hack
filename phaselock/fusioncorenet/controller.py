import numpy as np
from torusnet.mesh import SPECIALTIES

class RepairEvent:
    def __init__(self, event_type, target_coords, description):
        """
        event_type: "THROTTLE", "WAKE", "SLEEP", "REROUTE", "VERIFY"
        target_coords: (x, y) tuple or None
        description: string explanation
        """
        self.event_type = event_type
        self.target_coords = target_coords
        self.description = description

    def to_dict(self):
        return {
            "type": self.event_type,
            "target": f"tile_{self.target_coords[0]}_{self.target_coords[1]}" if self.target_coords else "global",
            "description": self.description
        }

class FusionCoreNet:
    def __init__(self, width=125, height=120, congestion_threshold=8, sensitivity=0.5, confinement=0.5):
        self.width = width
        self.height = height
        self.congestion_threshold = congestion_threshold
        self.sensitivity = sensitivity
        self.confinement = confinement
        self.repair_history = []

    def monitor(self, mesh, active_agents):
        """
        Monitors the TorusMesh tensor state and returns a dictionary of stability metrics,
        along with a list of generated RepairEvents.
        """
        queue_sizes = mesh.queue_size_array
        congested_mask = queue_sizes >= self.congestion_threshold
        congested_coords = np.argwhere(congested_mask)  # Coordinates of congested tiles: [y, x]
        
        # Active vs Sleeping
        total_tiles = self.width * self.height
        active_count = np.sum(mesh.state_array == 1)
        congested_count = np.sum(mesh.state_array == 2)
        sleeping_count = total_tiles - active_count - congested_count
        
        # Confidence and Consensus Metrics
        active_mask = mesh.state_array == 1
        active_confidences = mesh.confidence_array[active_mask]
        
        if len(active_confidences) > 0:
            mean_confidence = float(np.mean(active_confidences))
            std_confidence = float(np.std(active_confidences))
            consensus_score = mean_confidence * (1.0 - std_confidence)
        else:
            mean_confidence = 0.0
            std_confidence = 0.0
            consensus_score = 0.0
            
        # Message Spatial Entropy (measures concentration of work)
        total_messages = np.sum(queue_sizes)
        if total_messages > 0:
            p = queue_sizes.astype(np.float32) / total_messages
            p_safe = np.where(p > 0, p, 1.0)
            spatial_entropy = float(-np.sum(p * np.log2(p_safe)))
        else:
            spatial_entropy = 0.0

        max_entropy = np.log2(total_tiles)
        normalized_spatial_entropy = spatial_entropy / max_entropy if total_tiles > 1 else 1.0

        # --- Fusion Stability Parameters ---
        # P = local message pressure (total messages divided by active cores count)
        active_div = max(1, active_count)
        P_val = float(total_messages) / active_div
        
        # C = local routing capacity (WSE-3 colors capability per core)
        C_val = 12.0 + 24.0 * self.confinement
        
        # Beta = P / C
        beta = P_val / C_val
        
        # Epsilon (disagreement_score + congestion_score + overload_score)
        disagreement_score = 1.0 - consensus_score
        congestion_score = float(np.sum(queue_sizes >= self.congestion_threshold)) / total_tiles
        overload_score = float(np.sum(queue_sizes >= 12)) / total_tiles
        epsilon = disagreement_score + congestion_score + overload_score
        
        # q-like = circulation_score / loop_risk
        circulation_score = float(np.mean(mesh.messages_sent_array)) + 0.1
        loop_risk = float(np.std(mesh.queue_size_array)) + 0.1
        q_like = circulation_score / loop_risk

        metrics = {
            "active_count": int(active_count),
            "sleeping_count": int(sleeping_count),
            "congested_count": int(congested_count),
            "mean_confidence": mean_confidence,
            "std_confidence": std_confidence,
            "consensus_score": consensus_score,
            "total_messages": int(total_messages),
            "spatial_entropy": normalized_spatial_entropy,
            "beta": beta,
            "epsilon": epsilon,
            "q_like": q_like,
            "pressure": P_val,
            "capacity": C_val
        }

        # 5. Generate Repair Events and Wavelets
        repair_events = []
        
        # A. Trigger THROTTLE if Beta exceeds critical limits
        beta_critical = 0.25 - 0.20 * self.sensitivity # safety threshold governed by sensitivity
        if beta > beta_critical:
            # Find the most congested tile
            if len(congested_coords) > 0:
                y, x = int(congested_coords[0][0]), int(congested_coords[0][1])
                event = RepairEvent(
                    "THROTTLE",
                    (x, y),
                    f"Local beta {beta:.3f} > {beta_critical}. Throttling active sector around node ({x},{y})."
                )
                repair_events.append(event)
            else:
                event = RepairEvent(
                    "THROTTLE",
                    None,
                    f"Beta {beta:.3f} > critical threshold {beta_critical}. Reducing global packet generation."
                )
                repair_events.append(event)

        # B. Trigger REROUTE if congestion_score exceeds threshold
        congestion_threshold_ratio = 0.03 - 0.025 * self.sensitivity
        if congestion_score > congestion_threshold_ratio:
            # Find a congested tile
            if len(congested_coords) > 0:
                y, x = int(congested_coords[-1][0]), int(congested_coords[-1][1])
                event = RepairEvent(
                    "REROUTE",
                    (x, y),
                    f"Congestion ratio {congestion_score:.3f} exceeded limit. Rerouting traffic around node ({x},{y})."
                )
                repair_events.append(event)

        # C. Trigger VERIFY if disagreement_score exceeds threshold
        disagreement_threshold = 0.65
        if disagreement_score > disagreement_threshold and total_messages > 15:
            event = RepairEvent(
                "VERIFY",
                None,
                f"Disagreement {disagreement_score:.2f} too high. Calling verify clusters to resolve facts."
            )
            repair_events.append(event)

        # D. Trigger CONSENSUS if consensus_score is promising
        consensus_threshold = 0.50
        if consensus_score > consensus_threshold:
            event = RepairEvent(
                "CONSENSUS",
                None,
                f"Consensus score {consensus_score:.2f} is high. Directing Synthesizer agent to finalize."
            )
            repair_events.append(event)

        self.repair_history.extend(repair_events)
        if len(self.repair_history) > 100:
            self.repair_history = self.repair_history[-100:]

        return metrics, repair_events
