"""Describe existing input tails only. No model construction or training."""
from pathlib import Path
import json, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE.parents[1]
E = B.parent / 'cross_dataset_v1'
protocol = json.loads((B / 'protocol.json').read_text(encoding='utf8'))
records = []
for dataset in protocol['primary']:
    p = E / 'heldout_v1/prepared' / dataset / 'full'
    x = np.load(p / 'encoder_input.npy')
    counts = np.load(p / 'counts.npy')
    gene_support = np.count_nonzero(counts, axis=0)
    norms = np.linalg.norm(x.astype('float64'), axis=1)
    peak = np.unravel_index(np.argmax(x), x.shape)
    ids = np.load(p / 'cell_positions.npy')
    genes = np.load(p / 'gene_positions.npy')
    row = int(np.argmax(norms))
    run_path = B / 'runs' / dataset / 'scSAMAC/seed101/attempt_001/run.json'
    run = json.loads(run_path.read_text(encoding='utf8'))
    records.append(dict(
        dataset=dataset, n_cells=len(x), n_genes=x.shape[1],
        native_seed101_status=run['status'],
        genes_detected_in_one_cell=int((gene_support == 1).sum()),
        genes_detected_in_at_most_five_cells=int((gene_support <= 5).sum()),
        encoder_values_above_20=int((x > 20).sum()),
        encoder_values_above_50=int((x > 50).sum()),
        encoder_max=float(x[peak]), max_value_gene_support=int(gene_support[peak[1]]),
        max_value_raw_cell_position=int(ids[peak[0]]),
        max_value_raw_gene_position=int(genes[peak[1]]),
        max_row_l2=float(norms[row]), max_row_raw_cell_position=int(ids[row]),
        max_row_hvg_library=float(counts[row].sum()),
        max_row_detected_genes=int(np.count_nonzero(counts[row])),
        max_row_gene_values_above_50=int((x[row] > 50).sum()),
        median_row_l2=float(np.median(norms))))

(HERE / 'input_tail_diagnostic.json').write_text(json.dumps(dict(
    scope='Descriptive input statistics only; not evidence of the first failing model operation',
    training_started=False, optimizer_steps=0, records=records
), indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps(records, ensure_ascii=False, indent=2))
