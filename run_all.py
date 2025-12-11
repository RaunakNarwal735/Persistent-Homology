import subprocess
import argparse
import os
from datetime import datetime
import sys

def main():
    """ """
    parser = argparse.ArgumentParser(
        description="Run both persistence diagram and dynamic comparison for two networks"
    )
    parser.add_argument("--graphA", required=True, help="First network file (.kgml or .csv)")
    parser.add_argument("--graphB", required=True, help="Second network file (.kgml or .csv)")
    parser.add_argument("--nameA", default="A", help="Name label for graph A")
    parser.add_argument("--nameB", default="B", help="Name label for graph B")
    parser.add_argument("--outdir", default="outputs", help="Base output directory")
    parser.add_argument("--maxdim", type=int, default=1, help="Max homology dimension (for CRN.py)")
    parser.add_argument("--p", type=int, default=2, help="Wasserstein p-norm (for CRN.py)")
    parser.add_argument("--convert-kgml", action="store_true", help="Convert KGML to CSV as well")
    args = parser.parse_args()

   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n Running joint analysis")
    print(f"Results will be stored in: {run_dir}\n")

    
    print(">>> Running persistence diagram comparison...")
    crn_cmd = [
        sys.executable, "topology_compare.py",
        "--graphA", args.graphA,
        "--graphB", args.graphB,
        "--nameA", args.nameA,
        "--nameB", args.nameB,
        "--outdir", run_dir,
        "--maxdim", str(args.maxdim),
        "--p", str(args.p),
    ]
    if args.convert_kgml:
        crn_cmd.append("--convert-kgml")

    subprocess.run(crn_cmd, check=True)
    print(">>> Persistence diagram comparison completed.\n")

    
    print(">>> Running dynamic similarity analysis...")
    dyn_cmd = [
        sys.executable, "DC_plus.py",
        args.graphA,
        args.graphB,
        "--outdir", run_dir
    ]

    subprocess.run(dyn_cmd, check=True)
    print(">>> Dynamic comparison completed.\n")

    
    print(f"Outputs saved in: {run_dir}")

if __name__ == "__main__":
    main()
