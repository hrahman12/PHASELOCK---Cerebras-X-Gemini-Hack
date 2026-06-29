import numpy as np

# Define the 20 specialist AI agent specialties
SPECIALTIES = [
    "Planner",          # 0: Coordinates planning steps
    "Math",             # 1: Mathematical reasoning
    "Physics",          # 2: Physical systems and rules
    "Chemistry",        # 3: Molecular structures and bonds
    "Biology",          # 4: Life sciences and medicine
    "CodeGenerator",    # 5: Coding and syntax
    "Debugger",         # 6: Troubleshooting code issues
    "Verifier",         # 7: Verification and logical proof checking
    "Critic",           # 8: Peer reviews decisions and logic
    "MemoryRetriever",  # 9: Queries history and context
    "Linguistics",      # 10: Text parsing and grammatical syntax
    "Translator",       # 11: Inter-language translation
    "Summarizer",       # 12: Information compression
    "Synthesizer",      # 13: Answers aggregation
    "Logic",            # 14: Formal logic verification
    "Finance",          # 15: Quantitative finance and economics
    "Law",              # 16: Legal analysis and statutory checks
    "Psychology",       # 17: User intent and conversational alignment
    "Search",           # 18: Search query generation and analysis
    "FactChecker"       # 19: Truth and factual database checking
]

# Cerebras WSE-3 Hardware Specifications
CEREBRAS_WSE3_SPECS = {
    "total_cores": 900000,
    "wafer_grid_width": 990,
    "wafer_grid_height": 910,
    "clock_frequency_ghz": 1.1,
    "hop_latency_ns": 0.909,  # 1 / 1.1 GHz
    "bandwidth_gbps": 370.0,
    "die_size_mm": 215.0,
    "colors_count": 24  # WSE supports 24 fabric routing colors in hardware
}

class TorusMesh:
    def __init__(self, width=125, height=120, major_radius=3.0, minor_radius=1.0):
        self.width = width
        self.height = height
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.specs = CEREBRAS_WSE3_SPECS
        
        # State array: 0 = SLEEPING, 1 = ACTIVE, 2 = CONGESTED
        self.state_array = np.zeros((height, width), dtype=np.int32)
        
        # Confidence array: float value 0.0 to 1.0
        self.confidence_array = np.zeros((height, width), dtype=np.float32)
        
        # Queue sizes for incoming wavelets
        self.queue_size_array = np.zeros((height, width), dtype=np.int32)
        
        # Message counts (telemetry)
        self.messages_sent_array = np.zeros((height, width), dtype=np.int64)
        
        # Specialty ID mapping (0 to 19)
        self.specialty_array = np.zeros((height, width), dtype=np.int32)
        
        # Setup specialty distribution
        self.assign_specialties()
        
        # Compute coordinates
        self.coords_3d = self._compute_tokamak_coordinates()
        
    def assign_specialties(self):
        """
        Partitions the grid into 20 blocks (5 columns x 4 rows).
        Block width = 150 / 5 = 30 columns.
        Block height = 100 / 4 = 25 rows.
        This provides 20 distinct spatial regions of 750 tiles each.
        """
        cols = 5
        rows = 4
        block_w = self.width // cols
        block_h = self.height // rows
        
        for r in range(rows):
            for c in range(cols):
                specialty_id = r * cols + c
                y_start = r * block_h
                y_end = (r + 1) * block_h
                x_start = c * block_w
                x_end = (c + 1) * block_w
                self.specialty_array[y_start:y_end, x_start:x_end] = specialty_id

    def _compute_tokamak_coordinates(self):
        """
        Computes 3D Cartesian coordinates (X, Y, Z) for each tile
        on the surface of a torus representing Tokamak geometry.
        Shape: (height, width, 3)
        """
        Y, X = np.indices((self.height, self.width))
        
        theta = (2.0 * np.pi * Y) / self.height  # Poloidal angle
        phi = (2.0 * np.pi * X) / self.width    # Toroidal angle
        
        coords = np.zeros((self.height, self.width, 3), dtype=np.float32)
        coords[..., 0] = (self.major_radius + self.minor_radius * np.cos(theta)) * np.cos(phi) # X
        coords[..., 1] = (self.major_radius + self.minor_radius * np.cos(theta)) * np.sin(phi) # Y
        coords[..., 2] = self.minor_radius * np.sin(theta)                                      # Z
        
        return coords

    def get_neighbors(self, x, y):
        """
        Returns the 4 standard neighbors with torus-style periodic wrapping.
        """
        north = (x, (y - 1) % self.height)
        south = (x, (y + 1) % self.height)
        east = ((x + 1) % self.width, y)
        west = ((x - 1) % self.width, y)
        return {"north": north, "south": south, "east": east, "west": west}
        
    def get_helical_neighbors(self, x, y, pitch_x=2, pitch_y=1):
        """
        Returns the helical neighbors wrapping around the torus,
        mimicking magnetic field lines in a tokamak.
        """
        forward = ((x + pitch_x) % self.width, (y + pitch_y) % self.height)
        backward = ((x - pitch_x) % self.width, (y - pitch_y) % self.height)
        return {"forward": forward, "backward": backward}

    def reset_simulation_states(self):
        """
        Resets states for a new simulation run.
        """
        self.state_array.fill(0) # All sleeping
        self.confidence_array.fill(0.0)
        self.queue_size_array.fill(0)
        self.messages_sent_array.fill(0)
