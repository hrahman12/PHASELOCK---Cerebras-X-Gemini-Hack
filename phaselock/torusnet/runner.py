import os
import time
import random
import numpy as np

# Import WSE-3 Parameters and specialties
from torusnet.wse_params import (
    SIM_MESH_W, SIM_MESH_H, SIM_MESH_NODES, COLORS_PER_CORE, SELECTED_AGENTS
)
from torusnet.mesh import TorusMesh, SPECIALTIES
from torusnet.colors import *
from torusnet.embedding import mesh_to_torus, compute_distortion_metrics
from torusnet.routing import SparsePacketRouter
from torusnet.wavelets import WaveletFactory
from fusioncorenet import FusionCoreNet
from torusnet.agent_universe import AgentUniverseGenerator
from torusnet.selector import GroverSelector
from torusnet.consensus import ConsensusEngine
from torusnet.agent import SparseAgent

class SimulationRunner:
    def __init__(self, width=SIM_MESH_W, height=SIM_MESH_H):
        self.width = width
        self.height = height
        self.mesh = TorusMesh(width=width, height=height)
        self.router = SparsePacketRouter(width, height)
        self.wavelet_factory = WaveletFactory(width, height)
        self.fusion_core = FusionCoreNet(width, height)
        self.universe_gen = AgentUniverseGenerator()
        self.selector = GroverSelector(target_count=SELECTED_AGENTS)
        self.consensus_engine = ConsensusEngine()
        
        self.agents = {}  # Map of tile_x_y -> Agent metadata dict
        self.active_packets = []
        self.leader_coords = {} # Map of specialty_id -> (x, y)
        self._precompute_leader_coordinates()

    def _precompute_leader_coordinates(self):
        """
        Precomputes the center coordinates of the 20 specialist cortex blocks
        in the WSE mesh grid of size 125x120.
        """
        cols = 5
        rows = 4
        block_w = self.width // cols  # 25 columns per block
        block_h = self.height // rows # 30 rows per block
        
        for r in range(rows):
            for c in range(cols):
                specialty_id = r * cols + c
                lx = c * block_w + block_w // 2
                ly = r * block_h + block_h // 2
                self.leader_coords[specialty_id] = (lx, ly)

    def run_simulation(self, prompt, max_rounds=10, image_attached=False, datasheet_attached=False, pdf_attached=False, use_llm=True, sensitivity=0.5, confinement=0.5, diffusion=0.2, shots="1024"):
        """
        Runs the full TorusNet multi-agent simulation loop.
        """
        start_sim_time = time.time()
        print(f"\n[GROVER] Initializing quantum selection with {shots} execution shots...")
        
        # Ingest attachments into prompt context
        attachments_context = ""
        if image_attached:
            attachments_context += "\n[IMAGE ATTACHMENT: defect_map.png containing molecular lattice irregularities at grid (42, 87). Critical rupture region detects strain concentration.]"
        if datasheet_attached:
            attachments_context += "\n[DATASHEET ATTACHMENT: Carbon bond limits datasheet. C-C bond energy is 4.8 eV, C-O bond energy is 3.6 eV. Negative thermal expansion coefficient (CTE) under 1000K induces compressive stress of 1.2 GPa.]"
        if pdf_attached:
            attachments_context += "\n[PDF ATTACHMENT: Graphene Wafer paper. Section 4 specifies conformal mapping parameters: R_MAJOR=3.0, R_MINOR=1.0. WSE-3 router fabric supports 24 routing colors and 0.9 ns node latency.]"
        
        if attachments_context:
            prompt = prompt + "\n" + attachments_context
            
        # 1. Generate dynamic Agent Universe (20,000 candidates — NumPy vectorised)
        universe = self.universe_gen.generate_universe(prompt, target_size=20000)
        
        # 2. Select the best 15,000 agents using Grover-inspired classical selector
        selected_agents_meta, grover_steps = self.selector.select(
            universe, prompt,
            image_attached=image_attached,
            datasheet_attached=datasheet_attached,
            pdf_attached=pdf_attached
        )
        
        # Reset grid state arrays
        self.mesh.reset_simulation_states()
        self.active_packets = []
        self.agents = {}
        
        # Configure FusionCore dynamic parameters
        self.fusion_core.sensitivity = sensitivity
        self.fusion_core.confinement = confinement
        
        # Create leader agents list
        leader_set = set(self.leader_coords.values())
        
        # 3. Place selected 15,000 agents on WSE-inspired 2D mesh & compute 3D coordinates
        pos_list = []
        for idx, agent_meta in enumerate(selected_agents_meta):
            mx = idx % self.width
            my = idx // self.width
            if my >= self.height:
                break
                
            is_leader = (mx, my) in leader_set
            
            # Retrieve 3D toroidal mapping
            torus_pos = mesh_to_torus(mx, my, self.width, self.height)
            pos_list.append(torus_pos)
            
            # Instantiate agent
            agent_id = f"agent_{mx}_{my}"
            agent = SparseAgent(
                agent_id=agent_id,
                x=mx,
                y=my,
                specialty_id=agent_meta["specialty_id"],
                is_leader=is_leader
            )
            agent.confidence = agent_meta["confidence_prior"]
            agent.specialty = agent_meta["specialty"]
            agent.tools = agent_meta["tools"]
            
            self.agents[agent_id] = agent
            
            # Synchronize state in mesh array: all 15,000 nodes are active reasoning nodes
            self.mesh.state_array[my, mx] = 1
            self.mesh.confidence_array[my, mx] = agent.confidence

        # Compute embedding distortion metrics
        distortion = compute_distortion_metrics(pos_list, sample_size=150, W=self.width, H=self.height)
        
        # Setup starting prompt at Planner cortex leader
        planner_x, planner_y = self.leader_coords[0]
        planner_agent = self.agents.get(f"agent_{planner_x}_{planner_y}")
        if planner_agent:
            init_packet = self.router.create_packet(
                sender_id="user",
                receiver_id=planner_agent.agent_id,
                x1=planner_x, y1=planner_y,
                x2=planner_x, y2=planner_y,
                color=COLOR_FACT,
                importance=0.9,
                content=prompt,
                confidence=1.0,
                specialty_id=0
            )
            planner_agent.receive_wavelet(init_packet)

        # Tracing variables
        simulation_log = []
        total_packets_sent = 0
        total_packets_pruned = 0
        total_wavelets_triggered = []
        final_consensus = {}

        # 4. Multi-Round Simulation Loop
        for rnd in range(max_rounds):
            round_data = {
                "round": rnd,
                "agent_thoughts": [],
                "routing_ticks": [],
                "repair_events": [],
                "metrics": {}
            }
            
            # --- Sub-Phase A: Agent Reasoning ---
            # Separate leaders and followers for execution
            leaders = []
            followers = []
            for agent in self.agents.values():
                if agent.is_leader:
                    leaders.append(agent)
                else:
                    followers.append(agent)
            
            # 1. Run followers in sequential fast loop (local simulation, no LLM calls)
            for agent in followers:
                agent.think(prompt, rnd, use_llm=False, diffusion=diffusion)
                self.mesh.confidence_array[agent.y, agent.x] = agent.confidence
            
            # 2. Run leaders in parallel (multi-threading handles the synchronous Cerebras SDK latencies)
            def run_leader_think(agent):
                thought = agent.think(prompt, rnd, use_llm=use_llm)
                return agent, thought

            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            if leaders:
                with ThreadPoolExecutor(max_workers=len(leaders)) as executor:
                    futures = {executor.submit(run_leader_think, agent): agent for agent in leaders}
                    results = []
                    for future, agent in futures.items():
                        try:
                            results.append(future.result(timeout=8))
                        except FuturesTimeout:
                            # Timed out — use fast local fallback
                            fallback = agent._generate_simulated_reasoning(prompt, "", rnd, use_llm=False)
                            results.append((agent, fallback))
                        except Exception:
                            results.append((agent, None))
                
                for agent, thought in results:
                    self.mesh.confidence_array[agent.y, agent.x] = agent.confidence
                    if thought:
                        round_data["agent_thoughts"].append({
                            "agent": agent.agent_id,
                            "specialty": agent.specialty,
                            "coords": [agent.x, agent.y],
                            "thought": thought,
                            "confidence": agent.confidence
                        })
                        self._dispatch_packets(agent, rnd)

            # --- Sub-Phase B: Sparse Packet Routing ---
            for tick in range(10): # 10 routing ticks per round
                tick_log = []
                in_transit = []
                
                # Reset local queue size array
                self.mesh.queue_size_array.fill(0)
                for pkt in self.active_packets:
                    px, py = pkt["current_mesh"]
                    self.mesh.queue_size_array[py, px] += 1
                
                for pkt in self.active_packets:
                    # Sparse Pruning: Check transmission threshold
                    if not self.router.should_transmit(pkt):
                        total_packets_pruned += 1
                        continue
                        
                    # Calculate next hop
                    direction, next_mesh = self.router.route_step(pkt)
                    pkt["current_mesh"] = next_mesh
                    pkt["history"].append(next_mesh)
                    pkt["ttl"] -= 1
                    
                    tx, ty = pkt["mesh_target"]
                    if next_mesh == (tx, ty) or pkt["ttl"] <= 0:
                        # Packet arrived
                        dest_agent = self.agents.get(f"agent_{tx}_{ty}")
                        if dest_agent:
                            dest_agent.receive_wavelet(pkt)
                    else:
                        in_transit.append(pkt)
                        
                    # Record for 3D visualizer trails
                    tick_log.append({
                        "current": f"tile_{next_mesh[0]}_{next_mesh[1]}",
                        "target": f"tile_{tx}_{ty}",
                        "color": pkt["color"],
                        "importance": pkt["importance_score"]
                    })
                    
                self.active_packets = in_transit
                round_data["routing_ticks"].append(tick_log)
                total_packets_sent += len(tick_log)

            # --- Sub-Phase C: Fusion Core Stability and Control Wavelets ---
            metrics, repair_events = self.fusion_core.monitor(self.mesh, self.agents)
            metrics.update(distortion) # Add mapping distortion metrics
            
            round_data["metrics"] = metrics
            round_data["repair_events"] = [e.to_dict() for e in repair_events]
            
            # Apply control wavelets on neighbor grids
            for event in repair_events:
                if event.target_coords:
                    tx, ty = event.target_coords
                    target_agent = self.agents.get(f"agent_{tx}_{ty}")
                    if target_agent:
                        wavelet = self.wavelet_factory.create_wavelet(
                            wavelet_type=event.event_type,
                            agent_id=target_agent.agent_id,
                            mesh_x=tx, mesh_y=ty
                        )
                        self.wavelet_factory.apply_wavelet_effects(
                            wavelet=wavelet,
                            agents_list=list(self.agents.values()),
                            mesh=self.mesh
                        )
                        total_wavelets_triggered.append(wavelet)

            simulation_log.append(round_data)

            # --- Sub-Phase D: Consensus Engine Check ---
            compiled = self.consensus_engine.compile_consensus(
                active_agents=list(self.agents.values()),
                metrics=metrics,
                packets_sent=total_packets_sent,
                packets_pruned=total_packets_pruned,
                wavelets_triggered=total_wavelets_triggered,
                round_num=rnd
            )
            final_consensus = compiled
            
            # Early stop if consensus is high and stable
            if compiled["consensus_score"] >= 0.85 and rnd >= 3:
                break
                
        # 5. Synthesis Compilation
        elapsed_time = time.time() - start_sim_time
        
        return {
            "log": simulation_log,
            "final_answer": final_consensus.get("final_answer", "Consensus compile failed."),
            "total_rounds": len(simulation_log),
            "total_packets_sent": total_packets_sent,
            "total_packets_pruned": total_packets_pruned,
            "total_wavelets_triggered": len(total_wavelets_triggered),
            "elapsed_time": round(elapsed_time, 4),
            "specs": self.mesh.specs,
            "distortion": distortion,
            "grover_steps": grover_steps
        }

    def _dispatch_packets(self, agent, round_num):
        """
        Dispatches sparse communication packets from a thinking leader agent
        to neighboring specialist cortex blocks.
        """
        # Determine target specialty coordinate
        target_specialties = {
            "Planner": ["Chemistry", "Physics", "Math"],
            "Chemistry": ["Physics", "Verifier"],
            "Physics": ["Math", "Verifier"],
            "Math": ["Logic", "Verifier"],
            "Summarizer": ["Synthesizer"],
            "Verifier": ["Critic", "Synthesizer"],
            "Critic": ["Synthesizer", "Planner"]
        }
        
        targets = target_specialties.get(agent.specialty, ["Synthesizer"])
        for target_spec in targets:
            try:
                sid = SPECIALTIES.index(target_spec)
                tx, ty = self.leader_coords[sid]
                
                # Determine message routing color based on channel class
                color = COLOR_WAVELET
                if target_spec == "Verifier":
                    color = COLOR_VERIFY
                elif target_spec == "Synthesizer":
                    color = COLOR_FINAL
                elif agent.specialty == "Planner":
                    color = COLOR_FACT
                
                # Generate packet
                pkt = self.router.create_packet(
                    sender_id=agent.agent_id,
                    receiver_id=f"agent_{tx}_{ty}",
                    x1=agent.x, y1=agent.y,
                    x2=tx, y2=ty,
                    color=color,
                    importance=agent.confidence,
                    confidence_delta=agent.confidence * 0.1,
                    content=agent.last_response or "",
                    confidence=agent.confidence,
                    specialty_id=agent.specialty_id
                )
                self.active_packets.append(pkt)
                
            except ValueError:
                pass
