import argparse
import sys
from torusnet.runner import SimulationRunner
from torusnet.benchmark import BenchmarkSuite

def main():
    parser = argparse.ArgumentParser(description="TorusNet WSE Multi-Agent Simulator CLI")
    parser.add_argument("--prompt", type=str, default="What is the thermodynamic stability of a graphene monolayer under thermal strain?",
                        help="Prompt query for the agents")
    parser.add_argument("--rounds", type=int, default=8, help="Number of reasoning rounds")
    parser.add_argument("--benchmark", action="store_true", help="Run the topology benchmark comparison suite")
    args = parser.parse_args()

    print("======================================================================")
    print("TORUSNET: CEREBRAS-NATIVE MULTI-AGENT WSE SIMULATOR")
    print("======================================================================")

    if args.benchmark:
        print(f"\n[BENCHMARK] Running comparative suite on 15,000 logical tiles...")
        print("[BENCHMARK] Topologies: Ring, Star, Flat Mesh, Random, TorusNet\n")
        
        suite = BenchmarkSuite(width=125, height=120)
        results = suite.run_all(args.prompt)
        
        # Print comparison table
        print(f"{'Topology':<12} | {'Rounds':<6} | {'Avg Hops':<8} | {'Congestion':<10} | {'Cost':<6} | {'Consensus':<9}")
        print("-" * 65)
        for topo, stats in results.items():
            print(f"{topo:<12} | {stats['latency_rounds']:<6} | {stats['average_hops']:<8} | {stats['congestion_rate']:<9}% | {stats['communication_cost']:<6} | {stats['consensus_score']:<9}")
        print("-" * 65)
        print("\nNote: TorusNet uses wrapped torus coordinates combined with helical safety-factor paths")
        print("to minimize hops and routing congestion dynamically monitored by the Fusion Core.\n")
        return

    # Run standard simulation
    print(f"[INIT] Configuring 125x120 grid (15,000 agents)...")
    print(f"[INIT] Ingesting Prompt: '{args.prompt}'")
    print(f"[RUN] Starting simulation loop for {args.rounds} rounds...\n")

    runner = SimulationRunner(width=125, height=120)
    result = runner.run_simulation(args.prompt, max_rounds=args.rounds)

    # Print log steps
    for round_data in result["log"]:
        print(f"\n--- ROUND {round_data['round']} ---")
        for thought in round_data["agent_thoughts"]:
            print(f"  [{thought['specialty']} Leader ({thought['coords'][0]}, {thought['coords'][1]})]:")
            print(f"    \"{thought['thought']}\"")
        
        metrics = round_data["metrics"]
        print(f"  [METRICS] Active: {metrics['active_count']} | Congested: {metrics['congested_count']} | Messages: {metrics['total_messages']} | Consensus: {metrics['consensus_score']:.2%}")
        
        if round_data["repair_events"]:
            print("  [FUSION CORE REPAIR EVENTS]:")
            for event in round_data["repair_events"]:
                print(f"    * [{event['type']}] {event['description']}")

    print("\n======================================================================")
    print("FINAL CONVERGED ANSWER (SYNTHESIZED OUTPUT):")
    print("======================================================================")
    print(result["final_answer"])
    print("======================================================================")

if __name__ == "__main__":
    main()
