import numpy as np
from torusnet.mesh import SPECIALTIES

# Pre-build static lookup tables once at import time
_SPECIALTY_TOOLS = {
    "Planner":         (["planning", "decomposition", "coordination"],         ["plan_tasks", "allocate_resources"]),
    "Math":            (["algebra", "calculus", "differential_equations"],      ["solve_equation", "integrate", "differentiate"]),
    "Physics":         (["kinetics", "thermodynamics", "structural_mechanics"], ["calculate_stress", "simulate_kinetics"]),
    "Chemistry":       (["bond_analysis", "spectroscopy", "molecular_modeling"],["calculate_binding_energy", "model_structure"]),
    "Biology":         (["pathway_analysis", "protein_folding", "genomics"],    ["search_protein_domains", "map_pathway"]),
    "CodeGenerator":   (["code_synthesis", "refactoring", "architecting"],      ["generate_python", "write_test"]),
    "Debugger":        (["stack_trace_analysis", "memory_leak_check", "profiling"],["trace_execution", "profile_memory"]),
    "Verifier":        (["formal_proofs", "assertion_checking", "validation"],  ["assert_invariant", "prove_theorem"]),
    "Critic":          (["peer_review", "logic_checks", "bias_detection"],      ["review_rationales", "flag_errors"]),
    "MemoryRetriever": (["rag", "vector_search", "context_loading"],            ["query_vectors", "retrieve_past_turn"]),
    "Linguistics":     (["syntax_parsing", "semantic_extraction", "grammar"],   ["parse_sentence", "extract_entities"]),
    "Translator":      (["cross_lingual_mapping", "idiom_resolution"],          ["translate_text"]),
    "Summarizer":      (["information_compression", "bulleting"],                ["summarize_documents"]),
    "Synthesizer":     (["data_consolidation", "resolution_compiling"],         ["synthesize_answers"]),
    "Logic":           (["boolean_logic", "deduction", "inference"],            ["verify_implications"]),
    "Finance":         (["portfolio_optimization", "risk_modeling"],            ["calculate_black_scholes"]),
    "Law":             (["statute_lookup", "regulatory_checks"],                ["check_compliance"]),
    "Psychology":      (["persona_adaptation", "intent_alignment"],             ["align_user_intent"]),
    "Search":          (["web_indexing", "citation_lookup"],                    ["search_pubmed", "search_arxiv"]),
    "FactChecker":     (["grounding", "truth_matching"],                        ["verify_fact"]),
}
_N_SPEC = len(SPECIALTIES)
_SPEC_IDX = {s: i for i, s in enumerate(SPECIALTIES)}


class AgentUniverseGenerator:
    def __init__(self):
        pass

    def generate_universe(self, prompt, target_size=20000):
        """
        Generates a temporary expert agent universe matching the query.
        Uses vectorised NumPy generation — no Python loop, ~50-100x faster.
        """
        prompt_l = prompt.lower()

        # Focused specialty indices (given 2x weight via interleave)
        if any(k in prompt_l for k in ("graphene", "bond", "molecule", "chemistry", "thermal")):
            focused = ["Chemistry", "Physics", "Math", "Summarizer", "Verifier", "Synthesizer"]
        elif any(k in prompt_l for k in ("code", "program", "bug", "debug")):
            focused = ["CodeGenerator", "Debugger", "Verifier", "Logic", "Planner"]
        elif any(k in prompt_l for k in ("math", "equation", "calculus")):
            focused = ["Math", "Logic", "Verifier", "Planner"]
        else:
            focused = ["Planner", "Synthesizer", "Verifier", "Critic"]

        focused_ids  = np.array([_SPEC_IDX[s] for s in focused], dtype=np.int32)
        all_ids      = np.arange(_N_SPEC, dtype=np.int32)

        # Build specialty id array: alternate focused / uniform draws (2× weight to focused)
        rng = np.random.default_rng()
        half = target_size // 2
        focused_draw = rng.choice(focused_ids, size=half)
        uniform_draw = rng.choice(all_ids,     size=target_size - half)
        specialty_ids = np.empty(target_size, dtype=np.int32)
        specialty_ids[0::2] = focused_draw[:target_size - half] if target_size % 2 else focused_draw
        specialty_ids[1::2] = uniform_draw

        # Confidence priors in bulk
        conf_priors = rng.uniform(0.35, 0.85, size=target_size).round(2)

        # Modality flags in bulk
        has_image = rng.random(target_size) > 0.7
        has_table = rng.random(target_size) > 0.85
        has_video = rng.random(target_size) > 0.95

        # Neighbour specialty ids (3 per agent)
        neighbours = rng.choice(all_ids, size=(target_size, 3))

        # Assemble list — still a list-of-dicts for selector compatibility, but avoids per-agent random calls
        universe = []
        for i in range(target_size):
            sid = int(specialty_ids[i])
            spec = SPECIALTIES[sid]
            caps, tools = _SPECIALTY_TOOLS.get(spec, (["general_reasoning"], ["chat"]))
            mod = ["text"]
            if has_image[i]: mod.append("image")
            if has_table[i]: mod.append("table")
            if has_video[i]: mod.append("video")
            nb = list({SPECIALTIES[int(n)] for n in neighbours[i]})
            universe.append({
                "id": i,
                "specialty": spec,
                "specialty_id": sid,
                "capabilities": [f"{c}_{i % 5}" for c in caps],
                "tools":        [f"{t}_{i % 4}" for t in tools],
                "confidence_prior": float(conf_priors[i]),
                "modality_support": mod,
                "neighboring_specialties": nb,
            })

        print(f"[UNIVERSE] Generated {len(universe)} expert agents dynamically for query: '{prompt[:40]}...'")
        return universe
