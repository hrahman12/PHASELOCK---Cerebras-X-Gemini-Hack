import sys
import webbrowser
import threading
import time
from torusnet.server import start_server

def open_browser(url):
    # Wait a second for the server to spin up, then open browser
    time.sleep(1.0)
    print(f"Opening dashboard at {url} ...")
    webbrowser.open(url)

def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Defaulting to 8000.")

    url = f"http://localhost:{port}/"
    
    # Start browser opener in a separate thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    # Start the blocking server
    start_server(port)

if __name__ == "__main__":
    main()
