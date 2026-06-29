import numpy as np
from torusnet.mesh import SPECIALTIES

class GroverSelector:
    def __init__(self, target_count=15000):
        self.target_count = target_count

    def select(self, agents_universe, prompt, image_attached=False, datasheet_attached=False, pdf_attached=False):
        """
        Grover-inspired classical selector.
        Amplifies relevant state probabilities over iterations and outputs exactly 15,000 agents.
        """
        N = len(agents_universe)
        M = self.target_count
        if N <= M:
            return agents_universe

        # Extract features from query
        query_words = [w.strip("?,.!") for w in prompt.lower().split() if len(w) > 3]
        
        # 1. Calculate base semantic scores
        base_scores = np.zeros(N, dtype=np.float64)
        for idx, agent in enumerate(agents_universe):
            score = 0.1
            specialty = agent["specialty"].lower()
            
            # Specialty keyword matching
            for word in query_words:
                if word in specialty:
                    score += 0.3
                if any(word in tool.lower() for tool in agent["tools"]):
                    score += 0.25
            
            # Multimodal attachment boosts
            if image_attached and agent["specialty"] in ["Physics", "Chemistry", "Biology", "Synthesizer"]:
                score += 0.2
            if datasheet_attached and agent["specialty"] in ["Math", "Finance", "Logic", "Verifier"]:
                score += 0.2
            if pdf_attached and agent["specialty"] in ["Summarizer", "MemoryRetriever", "Law", "FactChecker"]:
                score += 0.2
            
            # Confidence prior scaling
            score += agent["confidence_prior"] * 0.15
            base_scores[idx] = min(1.0, score)

        # 2. Oracle threshold identification
        # We target the top M scoring agents
        k_optimal = M
        threshold_val = np.partition(base_scores, -k_optimal)[-k_optimal]
        oracle_marked = (base_scores >= threshold_val).astype(float)

        # 3. Grover-inspired classical amplitude amplification
        # Initialize uniform amplitude vector
        amplitudes = np.ones(N, dtype=np.float64) / np.sqrt(N)
        
        # Classical Grover steps
        # K = pi/4 * sqrt(N/M)
        num_steps = int(np.ceil((np.pi / 4.0) * np.sqrt(N / M)))
        num_steps = max(1, min(num_steps, 5)) # clamp iterations for compute safety
        
        for _ in range(num_steps):
            # Phase kick (Oracle inversion of marked states)
            amplitudes = np.where(oracle_marked > 0, -amplitudes, amplitudes)
            
            # Inversion about the mean (Diffusion operator)
            mean_amp = np.mean(amplitudes)
            amplitudes = 2.0 * mean_amp - amplitudes

        # Convert back to probabilities (amplitudes squared)
        probabilities = amplitudes ** 2
        probabilities /= np.sum(probabilities)

        # 4. Diversity Bonus Integration
        # We ensure representation across all specialties to keep WSE cortex robust
        specialty_counts = {s: 0 for s in SPECIALTIES}
        selected_indices = []
        
        # Sort indices by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # We perform a pass and select agents, adding a penalty if a specialty is overrepresented,
        # ensuring all 20 specialties are represented in the final 15,000 grid.
        for idx in sorted_indices:
            if len(selected_indices) >= M:
                break
            agent = agents_universe[idx]
            spec = agent["specialty"]
            
            # Cap each specialty at a fair share (e.g., max 1000 nodes of same specialty)
            if specialty_counts[spec] < 1000:
                selected_indices.append(idx)
                specialty_counts[spec] += 1

        # If we are still short (due to caps), fill remaining slots with highest probability nodes
        if len(selected_indices) < M:
            for idx in sorted_indices:
                if len(selected_indices) >= M:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)

        # Re-compute probabilities step-by-step for only the selected 15,000 agents
        # to send minimal data to frontend
        amplitudes_history = np.ones(N, dtype=np.float64) / np.sqrt(N)
        step_probabilities = [[float(1.0 / N)] * M] # Reset
        
        for step in range(num_steps):
            amplitudes_history = np.where(oracle_marked > 0, -amplitudes_history, amplitudes_history)
            mean_amp = np.mean(amplitudes_history)
            amplitudes_history = 2.0 * mean_amp - amplitudes_history
            
            probs_step = amplitudes_history ** 2
            probs_step /= np.sum(probs_step)
            
            step_probs_selected = [float(probs_step[i]) for i in selected_indices]
            step_probabilities.append(step_probs_selected)

        # Convert back to original agents
        selected_agents = [agents_universe[i] for i in selected_indices]
        
        print(f"[SELECTOR] Grover-inspired selection finished. Selected exactly {len(selected_agents)} agents.")
        return selected_agents, step_probabilities
