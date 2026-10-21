"""Comparación pareada de SpecAugment, usando únicamente validación."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use al menos dos semillas diferentes.')
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error('Revise epochs, batch-size y num-workers.')
    out = args.output_dir.resolve()
    # Un directorio nuevo evita mezclar configuraciones o ejecuciones incompletas.
    if not args.summarize_only:
        out.mkdir(parents=True, exist_ok=False)
    rows, fingerprints = [], set()
    for seed in args.seeds:
        for enabled in (False, True):
            variant = 'spec_on' if enabled else 'spec_off'
            run = out / f'{variant}_seed{seed}'
            if not args.summarize_only:
                command = [sys.executable, str(Path(__file__).with_name('trainmodels.py')),
                           '--phase', 'train', '--models', 'wav2vec',
                           '--dataset-dir', str(args.dataset_dir.resolve()),
                           '--output-dir', str(run), '--epochs', str(args.epochs),
                           '--batch-size', str(args.batch_size), '--seed', str(seed),
                           '--num-workers', str(args.num_workers),
                           '--wav2vec-spec-augment' if enabled else '--no-wav2vec-spec-augment']
                print(f'\nEjecutando {variant}, semilla {seed}', flush=True)
                subprocess.run(command, check=True)
            record = json.loads((run / 'validation/wav2vec.json').read_text(encoding='utf-8'))
            training = record['training_arguments']
            if (training['seed'] != seed or training['epochs'] != args.epochs
                    or training['batch_size'] != args.batch_size
                    or training['wav2vec_settings']['config']['apply_spec_augment'] != enabled):
                raise ValueError(f'Configuración inesperada: {run}')
            fingerprints.add(record['dataset_fingerprint'])
            rows.append(dict(variant=variant, seed=seed, best_epoch=record['best_epoch'],
                             **record['validation']))
    if len(fingerprints) != 1:
        raise ValueError('Las ejecuciones no utilizan el mismo corpus.')
    with (out / 'runs.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for variant in ('spec_off', 'spec_on'):
        values = [r['f1_macro'] for r in rows if r['variant'] == variant]
        summary.append(dict(variant=variant, seeds=len(values),
                            mean_f1_macro=statistics.mean(values),
                            std_f1_macro=statistics.stdev(values)))
    differences = [next(r['f1_macro'] for r in rows if r['seed'] == seed and r['variant'] == 'spec_on')
                   - next(r['f1_macro'] for r in rows if r['seed'] == seed and r['variant'] == 'spec_off')
                   for seed in args.seeds]
    result = dict(summary=summary, paired_differences_on_minus_off=differences,
                  mean_paired_difference=statistics.mean(differences),
                  note='Resultados descriptivos de validación; no prueban significación estadística.')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
