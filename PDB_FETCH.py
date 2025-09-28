import os
import requests
import gzip
import pandas as pd
import io
import argparse
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

def fetch_pdb_ids(resolution_cutoff=2.5):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution_cutoff
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True
        }
    }
    res = requests.post(url, json=query)
    res.raise_for_status()
    return [entry["identifier"] for entry in res.json()["result_set"]]

def download_and_parse(pdb_id):
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        r = requests.get(url)
        if r.status_code != 200:
            return []

        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            mmcif = MMCIF2Dict(f)

        r_free = mmcif.get("_refine.ls_R_factor_R_free")
        if isinstance(r_free, list):
            r_free = r_free[0]
        if r_free is not None:
            r_free = str(r_free).strip()

        resolution = mmcif.get("_refine.ls_d_res_high")
        if isinstance(resolution, list):
            resolution = resolution[0]
        if resolution is not None:
            resolution = str(resolution).strip()

        poly_types = mmcif.get("_entity_poly.type")
        sequences = mmcif.get("_entity_poly.pdbx_seq_one_letter_code_can")
        chain_lists = mmcif.get("_entity_poly.pdbx_strand_id")

        results = []
        if poly_types and sequences and chain_lists:
            if isinstance(poly_types, str):
                poly_types = [poly_types]
            if isinstance(sequences, str):
                sequences = [sequences]
            if isinstance(chain_lists, str):
                chain_lists = [chain_lists]
            for t, s, c_list in zip(poly_types, sequences, chain_lists):
                if t.strip() == 'polypeptide(L)':
                    seq = s.replace('\n', '').strip()
                    seq_len = len(seq)
                    for chain_id in c_list.replace(' ', '').split(','):
                        if seq and seq_len > 50 and 'U' not in seq:
                            results.append((pdb_id, chain_id, seq, seq_len, r_free, resolution))
        return results
    except Exception as e:
        print(f"Failed {pdb_id}: {e}")
        return []

def filter_results(results):
    filtered = []
    for pdb_id, chain_id, seq, seq_len, r_free, resolution in results:
        try:
            if r_free is None or r_free == '?' or resolution is None or resolution == '?':
                continue
            r_free_val = float(r_free)
            resolution_val = float(resolution)
            if r_free_val < 0.25 and resolution_val <= 2.5:
                filtered.append((pdb_id, chain_id, seq, seq_len, r_free_val, resolution_val))
        except Exception:
            continue
    return filtered

def write_csv(filtered_results, output_file):
    df = pd.DataFrame(filtered_results, columns=["PDB_ID", "CHAIN_ID", "PROTEIN_SEQUENCE", "SEQ_LENGTH", "R_FREE", "RESOLUTION"])
    df.to_csv(output_file, index=False)

def main(workers=None, batch_size=2000, resolution_cutoff=2.5):
    if workers is None:
        workers = min(cpu_count() * 8, 32)
    print(f"Using {workers} workers")

    pdb_ids = fetch_pdb_ids(resolution_cutoff=resolution_cutoff)
    print(f"Fetched {len(pdb_ids)} PDB IDs with resolution ≤ {resolution_cutoff} Å")

    all_results = []
    batch_num = 1
    for i in range(0, len(pdb_ids), batch_size):
        batch_ids = pdb_ids[i:i+batch_size]
        checkpoint_file = f"filtered_protein_chains_batch_{batch_num}.csv"
        if os.path.exists(checkpoint_file):
            print(f"Skipping batch {batch_num}, already exists")
            batch_num += 1
            continue
        batch_results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_and_parse, pdb_id): pdb_id for pdb_id in batch_ids}
            for f in tqdm(as_completed(futures), total=len(batch_ids), desc=f"Batch {batch_num}"):
                batch_results.extend(f.result())
        filtered = filter_results(batch_results)
        write_csv(filtered, checkpoint_file)
        print(f"Saved {len(filtered)} protein chains to {checkpoint_file}")
        all_results.extend(filtered)
        batch_num += 1
    print(f"All done. Total filtered chains: {len(all_results)}")

    # Auto-download CSVs in Google Colab
    try:
        from google.colab import files
        for i in range(1, batch_num):
            file_name = f"filtered_protein_chains_batch_{i}.csv"
            if os.path.exists(file_name):
                files.download(file_name)
    except ImportError:
        print("Not running in Google Colab, skipping auto-download.")

if __name__ == "__main__":
    main()
