import os
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from torusnet.runner import SimulationRunner
from torusnet.benchmark import BenchmarkSuite
from torusnet.mesh import TorusMesh, SPECIALTIES

class TorusNetRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        # API Endpoints
        if path == "/api/layout":
            self.handle_api_layout()
        elif path == "/api/run":
            self.handle_api_run(query)
        elif path == "/api/benchmark":
            self.handle_api_benchmark(query)
        else:
            # Serve static files from the 'web' directory
            # If path is '/' serve index.html
            if path == "/":
                local_path = os.path.join("web", "index.html")
            else:
                # Strip leading slash and find file inside 'web/'
                rel_path = path.lstrip("/")
                local_path = os.path.join("web", rel_path)
            
            # Prevent directory traversal attacks
            abs_web = os.path.abspath("web")
            abs_local = os.path.abspath(local_path)
            
            if os.path.exists(local_path) and os.path.isfile(local_path) and abs_local.startswith(abs_web):
                self.send_response(200)
                mime_type, _ = mimetypes.guess_type(local_path)
                self.send_header("Content-Type", mime_type or "text/plain")
                self.end_headers()
                with open(local_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 Not Found")

    def handle_api_layout(self):
        """
        Returns the compact 15,000 WSE node coordinates, specialties,
        and leader flags in JSON format.
        """
        mesh = TorusMesh(width=125, height=120)
        
        # Precompute the coordinate leaders mapping
        cols = 5
        rows = 4
        block_w = mesh.width // cols
        block_h = mesh.height // rows
        leader_coords = set()
        for r in range(rows):
            for c in range(cols):
                lx = c * block_w + block_w // 2
                ly = r * block_h + block_h // 2
                leader_coords.add((lx, ly))

        nodes_data = []
        for y in range(mesh.height):
            for x in range(mesh.width):
                sid = int(mesh.specialty_array[y, x])
                is_leader = 1 if (x, y) in leader_coords else 0
                cx, cy, cz = mesh.coords_3d[y, x]
                # Flat list format: [x, y, specialty_id, is_leader, X, Y, Z]
                nodes_data.append([x, y, sid, is_leader, float(cx), float(cy), float(cz)])

        response = {
            "width": mesh.width,
            "height": mesh.height,
            "specialties": SPECIALTIES,
            "nodes": nodes_data,
            "specs": mesh.specs
        }
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def handle_api_run(self, query):
        """
        Runs the simulation for a given prompt and returns full execution history logs.
        """
        prompt = query.get("prompt", ["Analyze the molecular stability of a graphene-oxide monolayer under thermal strain."])[0]
        
        image_attached = query.get("image_attached", ["false"])[0] == "true"
        datasheet_attached = query.get("datasheet_attached", ["false"])[0] == "true"
        pdf_attached = query.get("pdf_attached", ["false"])[0] == "true"
        
        # Parse FusionCoreNet control knobs
        sensitivity = float(query.get("sensitivity", ["0.5"])[0])
        confinement = float(query.get("confinement", ["0.5"])[0])
        diffusion = float(query.get("diffusion", ["0.2"])[0])
        shots = query.get("shots", ["1024"])[0]
        
        runner = SimulationRunner(width=125, height=120)
        sim_res = runner.run_simulation(
            prompt,
            max_rounds=1,
            image_attached=image_attached,
            datasheet_attached=datasheet_attached,
            pdf_attached=pdf_attached,
            sensitivity=sensitivity,
            confinement=confinement,
            diffusion=diffusion,
            shots=shots
        )
        
        # Automatically run background benchmark comparison
        suite = BenchmarkSuite(width=125, height=120)
        benchmark_results = suite.run_all(prompt)
        sim_res["benchmark_comparison"] = benchmark_results
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(sim_res).encode("utf-8"))

    def handle_api_benchmark(self, query):
        """
        Runs the benchmark suite comparing TorusNet against other topologies.
        """
        prompt = query.get("prompt", ["Standard test benchmark query"])[0]
        
        suite = BenchmarkSuite(width=150, height=100)
        benchmark_results = suite.run_all(prompt)
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(benchmark_results).encode("utf-8"))

def start_server(port=8000):
    os.makedirs("web", exist_ok=True)
    server_address = ("", port)
    httpd = HTTPServer(server_address, TorusNetRequestHandler)
    print(f"TorusNet Web Server running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TorusNet Server.")
        httpd.server_close()
