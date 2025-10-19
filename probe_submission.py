import pandas as pd
import numpy as np
import argparse
import os

def run_probe(input_file: str, output_file: str, num_blocks: int, invert_block: int = -1, invert_all: bool = False):

    print(f"Reading submission file from: {input_file}")
    df = pd.read_csv(input_file)
 
    df['rule_violation'] = pd.to_numeric(df['rule_violation'])
    
    n_rows = len(df)
    print(f"Total rows: {n_rows}")

    if invert_all:
        print("Mode: Invert All. Transforming p -> 1-p for all predictions.")
        df['rule_violation'] = 1 - df['rule_violation']
    
    elif invert_block > 0:
        if not (1 <= invert_block <= num_blocks):
            raise ValueError(f"invert_block must be between 1 and {num_blocks}")
            
        print(f"Mode: Invert Gain Probe. Testing block {invert_block}/{num_blocks}.")
        print("Step 1: Inverting all predictions (p -> 1-p).")
        df['rule_violation'] = 1 - df['rule_violation']
        
        # 计算区块的边界
        block_size = n_rows / num_blocks
        start_index = int((invert_block - 1) * block_size)
        end_index = int(invert_block * block_size) if invert_block < num_blocks else n_rows
        
        print(f"Step 2: Reverting block {invert_block} (indices {start_index} to {end_index-1}) back to original (1-p -> p).")
        
        df.iloc[start_index:end_index, df.columns.get_loc('rule_violation')] = 1 - df.iloc[start_index:end_index]['rule_violation']

    else:
        print("Mode: Standard copy. No transformation applied.")

    print(f"Saving probed submission to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Submission Probing Tool")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the baseline submission CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the new submission CSV file.")
    parser.add_argument("--num_blocks", type=int, default=10, help="Total number of blocks to split the test set into.")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--invert_all", action="store_true", help="Invert all predictions (p -> 1-p).")
    mode_group.add_argument("--invert_block", type=int, default=-1, help="The block number (1-based) to revert back to original after a full inversion.")

    args = parser.parse_args()
    
    run_probe(args.input_file, args.output_file, args.num_blocks, args.invert_block, args.invert_all)
