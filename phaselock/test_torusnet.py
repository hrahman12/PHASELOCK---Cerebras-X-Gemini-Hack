import unittest
import numpy as np

from torusnet.wse_params import SIM_MESH_W, SIM_MESH_H, SELECTED_AGENTS
from torusnet.mesh import TorusMesh
from torusnet.embedding import mesh_to_torus, calculate_mesh_distance, calculate_torus_distance, compute_distortion_metrics
from torusnet.routing import SparsePacketRouter
from torusnet.wavelets import WaveletFactory
from fusioncorenet import FusionCoreNet
from torusnet.agent_universe import AgentUniverseGenerator
from torusnet.selector import GroverSelector
from torusnet.consensus import ConsensusEngine
from torusnet.agent import SparseAgent

class TestTorusNetUpdate(unittest.TestCase):
    def setUp(self):
        self.width = 125
        self.height = 120
        self.mesh = TorusMesh(self.width, self.height)
        self.router = SparsePacketRouter(self.width, self.height)
        self.wavelet_factory = WaveletFactory(self.width, self.height)
        self.fc = FusionCoreNet(self.width, self.height)

    def test_wse_simulation_dimensions(self):
        self.assertEqual(SIM_MESH_W, 125)
        self.assertEqual(SIM_MESH_H, 120)
        self.assertEqual(SELECTED_AGENTS, 15000)

    def test_embedding_coordinates(self):
        # Test mapping of coordinate (0, 0)
        pos0 = mesh_to_torus(0, 0, self.width, self.height)
        self.assertEqual(pos0["mesh_x"], 0)
        self.assertEqual(pos0["mesh_y"], 0)
        self.assertAlmostEqual(pos0["phi"], 0.0)
        self.assertAlmostEqual(pos0["theta"], 0.0)
        
        # At (0,0), X = (R_MAJOR + R_MINOR) = 4.0, Y = 0.0, Z = 0.0
        self.assertAlmostEqual(pos0["X"], 4.0)
        self.assertAlmostEqual(pos0["Y"], 0.0)
        self.assertAlmostEqual(pos0["Z"], 0.0)

    def test_distortion_calculations(self):
        pos_list = [
            mesh_to_torus(0, 0, self.width, self.height),
            mesh_to_torus(1, 0, self.width, self.height),
            mesh_to_torus(0, 1, self.width, self.height),
            mesh_to_torus(2, 2, self.width, self.height)
        ]
        
        # Test mesh distance calculations
        d_mesh = calculate_mesh_distance(0, 0, 2, 2, self.width, self.height)
        self.assertEqual(d_mesh, 4)
        
        # Test wrapped mesh distance (e.g. 0 to width-1 should be 1 hop)
        d_mesh_wrap = calculate_mesh_distance(0, 0, self.width - 1, 0, self.width, self.height)
        self.assertEqual(d_mesh_wrap, 1)

    def test_sparse_routing(self):
        # Create packet
        pkt = self.router.create_packet(
            sender_id="agent_0_0",
            receiver_id="agent_5_5",
            x1=0, y1=0,
            x2=5, y2=5,
            color=1,
            importance=0.8
        )
        
        self.assertEqual(pkt["mesh_source"], (0, 0))
        self.assertEqual(pkt["mesh_target"], (5, 5))
        self.assertTrue(self.router.should_transmit(pkt))
        
        # Check pruning threshold
        pruned_pkt = self.router.create_packet("a", "b", 0, 0, 5, 5, 1, importance=0.1)
        self.assertFalse(self.router.should_transmit(pruned_pkt))

    def test_routing_dor_steps(self):
        # Route packet from (0,0) to (2, 2)
        pkt = self.router.create_packet("a", "b", 0, 0, 2, 2, 1)
        
        # Step 1: Should go East along X dimension (0,0) -> (1, 0)
        dir1, next1 = self.router.route_step(pkt)
        self.assertEqual(dir1, "E")
        self.assertEqual(next1, (1, 0))
        
        # Update current mesh coordinate in packet and route again
        pkt["current_mesh"] = next1
        dir2, next2 = self.router.route_step(pkt)
        self.assertEqual(dir2, "E")
        self.assertEqual(next2, (2, 0))
        
        # X matches, step South along Y dimension (2, 0) -> (2, 1)
        pkt["current_mesh"] = next2
        dir3, next3 = self.router.route_step(pkt)
        self.assertEqual(dir3, "S")
        self.assertEqual(next3, (2, 1))

    def test_agent_universe_generation(self):
        gen = AgentUniverseGenerator()
        universe = gen.generate_universe("Graphene monolayer", target_size=100)
        self.assertEqual(len(universe), 100)
        self.assertEqual(universe[0]["id"], 0)
        self.assertTrue(isinstance(universe[0]["specialty"], str))
        self.assertTrue(len(universe[0]["tools"]) > 0)

    def test_grover_selector(self):
        gen = AgentUniverseGenerator()
        universe = gen.generate_universe("Analyze chemistry and thermal strain", target_size=200)
        
        selector = GroverSelector(target_count=50)
        selected, step_probs = selector.select(universe, "graphene molecular structure")
        
        self.assertEqual(len(selected), 50)
        self.assertTrue(len(step_probs) > 0)
        # Verify diversity representation: selected agents shouldn't be only one type
        specialties = set(a["specialty"] for a in selected)
        self.assertGreater(len(specialties), 1)

    def test_fusion_stability_monitoring(self):
        # Set up a simulation scenario in the mesh
        self.mesh.state_array.fill(0)
        self.mesh.queue_size_array.fill(0)
        
        # Congest one node
        self.mesh.queue_size_array[10, 10] = 15
        self.mesh.state_array[10, 10] = 1 # Active
        self.mesh.confidence_array[10, 10] = 0.8
        
        active_agents = {
            "agent_10_10": SparseAgent("agent_10_10", 10, 10, 1, is_leader=True)
        }
        
        metrics, repair_events = self.fc.monitor(self.mesh, active_agents)
        
        # Verify calculated fusion metrics
        self.assertTrue("beta" in metrics)
        self.assertTrue("epsilon" in metrics)
        self.assertTrue("q_like" in metrics)
        
        # With active_count=1 and messages=15:
        # Pressure = 15.0, Capacity = 24.0 -> Beta = 15.0 / 24.0 = 0.625
        self.assertAlmostEqual(metrics["beta"], 0.625)
        
        # Should trigger THROTTLE because beta > beta_critical (0.15)
        event_types = [e.event_type for e in repair_events]
        self.assertIn("THROTTLE", event_types)

    def test_consensus_engine(self):
        engine = ConsensusEngine()
        metrics = {"consensus_score": 0.38, "epsilon": 1.2} # High disagreement (epsilon/3 = 0.40)
        
        active_agents = [
            SparseAgent("a1", 0, 0, 1),
            SparseAgent("a2", 1, 1, 2)
        ]
        active_agents[0].specialty = "Planner"
        active_agents[0].last_response = "Ephemeral state"
        active_agents[0].confidence = 0.85
        active_agents[0].state = "ACTIVE"
        
        active_agents[1].specialty = "Synthesizer"
        active_agents[1].last_response = "Synthesized response"
        active_agents[1].confidence = 0.90
        active_agents[1].state = "ACTIVE"
        
        res = engine.compile_consensus(
            active_agents=active_agents,
            metrics=metrics,
            packets_sent=100,
            packets_pruned=50,
            wavelets_triggered=[],
            round_num=2
        )
        
        # Disagreement is low, so final_answer should be the Synthesizer response
        self.assertFalse(res["arbiter_activated"])
        self.assertEqual(res["final_answer"], "Synthesized response")
        
        # If disagreement is high (epsilon/3 = 0.6) and round_num is 5, Arbiter should be activated
        high_disagreement_metrics = {"consensus_score": 0.20, "epsilon": 1.8}
        res_arbiter = engine.compile_consensus(
            active_agents=active_agents,
            metrics=high_disagreement_metrics,
            packets_sent=100,
            packets_pruned=50,
            wavelets_triggered=[],
            round_num=5
        )
        self.assertTrue(res_arbiter["arbiter_activated"])
        self.assertTrue("ARBITER RESOLUTION" in res_arbiter["final_answer"])

if __name__ == "__main__":
    unittest.main()
