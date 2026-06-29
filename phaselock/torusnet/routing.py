import time
import random
from torusnet.embedding import mesh_to_torus

IMPORTANCE_THRESHOLD = 0.25

class SparsePacketRouter:
    def __init__(self, W=125, H=120):
        self.W = W
        self.H = H

    def create_packet(self, sender_id, receiver_id, x1, y1, x2, y2, color, importance=0.5, confidence_delta=0.0, content="", confidence=0.5, specialty_id=0):
        """
        Creates a WSE-3 style 32-bit sparse packet.
        """
        src_torus = mesh_to_torus(x1, y1, self.W, self.H)
        tgt_torus = mesh_to_torus(x2, y2, self.W, self.H)
        
        return {
            "packet_bits": 32,
            "data_bits": 16,
            "control_bits": 16,
            "color": color,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "mesh_source": (x1, y1),
            "mesh_target": (x2, y2),
            "torus_source": (src_torus["X"], src_torus["Y"], src_torus["Z"]),
            "torus_target": (tgt_torus["X"], tgt_torus["Y"], tgt_torus["Z"]),
            "current_mesh": (x1, y1),
            "importance_score": round(importance, 3),
            "confidence_delta": round(confidence_delta, 3),
            "ttl": 32, # max hops
            "timestamp": time.time(),
            "history": [(x1, y1)],
            "content": content,
            "confidence": confidence,
            "specialty_id": specialty_id
        }

    def route_step(self, packet):
        """
        Calculates the next coordinate for the packet using 5-port Dimension-Order Routing.
        Directions: N, S, E, W, CORE
        """
        cx, cy = packet["current_mesh"]
        tx, ty = packet["mesh_target"]
        
        if (cx, cy) == (tx, ty):
            return "CORE", (cx, cy)
            
        # Wrap-around aware routing
        # East/West routing along X dimension
        dx = (tx - cx) % self.W
        if dx != 0:
            # Check if moving East or West is shorter
            if dx <= self.W // 2:
                next_x = (cx + 1) % self.W
                direction = "E"
            else:
                next_x = (cx - 1) % self.W
                direction = "W"
            return direction, (next_x, cy)
            
        # North/South routing along Y dimension
        dy = (ty - cy) % self.H
        if dy != 0:
            # Check if moving South or North is shorter
            if dy <= self.H // 2:
                next_y = (cy + 1) % self.H
                direction = "S"
            else:
                next_y = (cy - 1) % self.H
                direction = "N"
            return direction, (cx, next_y)
            
        return "CORE", (cx, cy)

    def should_transmit(self, packet):
        """
        Applies sparse communication pruning: drop packets with low importance score.
        """
        return packet["importance_score"] >= IMPORTANCE_THRESHOLD
