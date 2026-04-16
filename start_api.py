"""start_api.py — Start the FastAPI server."""

import argparse, uvicorn, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default=API_HOST)
    parser.add_argument("--port",   type=int, default=API_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    print(f"\n🚀 Service Catalog RAG API (LangChain)")
    print(f"   URL  : http://{args.host}:{args.port}")
    print(f"   Docs : http://{args.host}:{args.port}/docs\n")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)
