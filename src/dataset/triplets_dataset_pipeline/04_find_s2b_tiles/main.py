import argparse
import sys
import os

from functions import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel-2 Batch Sourcing Pipeline for PhiSat-2")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input CSV file containing PhiSat-2 coordinates and metadata."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to the output CSV file where matched Sentinel-2 results will be appended."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Critical Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print("==================================================")
    print("Initiating Sentinel-2 Sourcing Pipeline")
    print(f"Input Data:  {args.input}")
    print(f"Output Data: {args.output}")
    print("==================================================")

    try:
        process_dataset(args.input, args.output)
        print("\n==================================================")
        print("Pipeline execution finished successfully.")
        print("==================================================")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been safely saved to the output CSV.")
        sys.exit(0)
    except Exception as e:
        print(f"\nPipeline crashed with a critical error: {e}")
        sys.exit(1)