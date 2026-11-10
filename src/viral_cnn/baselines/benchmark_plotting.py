"""M02 benchmark tables and figures derived from the persisted tabular evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from viral_cnn.baselines.paired import evaluate_predictions, wilson_score_interval
from viral_cnn.utils.logging_utils import log_artifact_path

STRATA=('all','overlap','no_exact_overlap')


def benchmark_summary(truth: np.ndarray, predictions: dict[str,dict[float,np.ndarray]],
                      masks: dict[str,np.ndarray]) -> pd.DataFrame:
    """Preserve every method/rate/stratum and all abstentions in its denominator."""
    rows=[]
    for method,rates in predictions.items():
        for rate,pred in rates.items():
            for stratum in STRATA:
                mask=masks[stratum];metrics=evaluate_predictions(truth[mask],pred[mask])
                low,high=wilson_score_interval(metrics['n_covered'],metrics['n'])
                rows.append({'method':method,'noise_rate':rate,'stratum':stratum,**metrics,
                    'n_species':metrics['n_classes'],'coverage_ci_low':low,'coverage_ci_high':high,
                    'ci_method':'Wilson','ci_level':.95,'noise_realizations':1})
    return pd.DataFrame(rows)


def _table(path: str | Path) -> tuple[pd.DataFrame,str]:
    log_artifact_path('M02 figure source CSV',path)
    return pd.read_csv(path),hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_curves(source_csv: str | Path, output_dir: str | Path) -> list[dict[str,Any]]:
    """Render seven-method accuracy/coverage curves in three strata from a CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df,digest=_table(source_csv);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for metric in ('accuracy','coverage'):
        fig,axes=plt.subplots(1,3,figsize=(19,5),sharey=True)
        for ax,stratum in zip(axes,STRATA):
            sub=df[df.stratum==stratum]
            for method in df.method.unique():
                rows=sub[sub.method==method].sort_values('noise_rate')
                ax.plot(rows.noise_rate,rows[metric],marker='o',label=method)
                ax.fill_between(rows.noise_rate.to_numpy(),rows[metric+'_ci_low'].to_numpy(),
                                rows[metric+'_ci_high'].to_numpy(),alpha=.08)
            n=int(sub.n.iloc[0]) if len(sub) else 0
            ax.set(title=f'{stratum} | N={n} | Wilson 95%',xlabel='Nominal noise ε',ylim=(0,1.05))
            ax.grid(alpha=.25)
        axes[0].set_ylabel(metric.replace('_',' ').title());axes[-1].legend(fontsize=7)
        fig.suptitle('One noise realization per rate; conditional on observed fragments')
        fig.tight_layout();record={'metric':metric,'source_csv':str(Path(source_csv).resolve()),'source_sha256':digest}
        for ext in ('png','pdf'):
            path=out/f'benchmark_{metric}.{ext}';fig.savefig(path,dpi=180);record[ext]=str(path.resolve())
        plt.close(fig);records.append(record)
    return records


def paired_differences(truth: np.ndarray, predictions: dict[str,dict[float,np.ndarray]],
                       masks: dict[str,np.ndarray]) -> pd.DataFrame:
    """Export all ten CNN/tool pairs using the existing C03 species bootstrap."""
    from viral_cnn.baselines.paired import bootstrap_species_difference
    rows=[]
    cnns=[m for m in predictions if m not in ('Kraken2-flat','BLASTN')]
    for a in cnns:
        for b in ('Kraken2-flat','BLASTN'):
            for rate,pa in predictions[a].items():
                for stratum in STRATA:
                    mask=masks[stratum];y=truth[mask]
                    diff=(pa[mask]==y).astype(float)-(predictions[b][rate][mask]==y).astype(float)
                    species=len(np.unique(y));reason='empty_stratum' if not len(y) else 'fewer_than_two_species' if species<2 else ''
                    low,high=bootstrap_species_difference(diff,y,repeats=2000,seed=42) if not reason else (np.nan,np.nan)
                    rows.append({'method_a':a,'method_b':b,'noise_rate':rate,'stratum':stratum,
                        'n':len(y),'n_species':species,'diff_mean_pp':100*float(diff.mean()) if len(y) else np.nan,
                        'ci_lower_pp':low,'ci_upper_pp':high,'ci_status':'unavailable' if reason else 'available',
                        'ci_reason':reason,'resampling_unit':'species','repeats':2000,'seed':42,
                        'difference_unit':'percentage_points','ci_level':.95,'noise_realizations':1})
    return pd.DataFrame(rows)


def render_differences(source_csv: str | Path, output_dir: str | Path) -> dict[str,Any]:
    """Render all paired deltas in strata panels, preserving points without ICs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df,digest=_table(source_csv);fig,axes=plt.subplots(1,3,figsize=(19,5),sharey=True)
    for ax,stratum in zip(axes,STRATA):
        sub=df[df.stratum==stratum]
        for (a,b),rows in sub.groupby(['method_a','method_b'],sort=False):
            rows=rows.sort_values('noise_rate')
            ax.plot(rows.noise_rate,rows.diff_mean_pp,marker='.',label=f'{a} − {b}')
            # Vertical bounds avoid assuming percentile intervals contain the point estimate.
            ax.vlines(rows.noise_rate,rows.ci_lower_pp,rows.ci_upper_pp,alpha=.25)
        n=int(sub.n.iloc[0]) if len(sub) else 0
        ax.axhline(0,color='gray',linestyle='--');ax.set(title=f'{stratum} | N={n}',xlabel='Nominal noise ε');ax.grid(alpha=.25)
    axes[0].set_ylabel('CNN − tool accuracy (percentage points)');axes[-1].legend(fontsize=6)
    fig.suptitle('Paired species bootstrap 95%; 2,000 repetitions; seed 42; one noise realization')
    fig.tight_layout();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    record={'source_csv':str(Path(source_csv).resolve()),'source_sha256':digest}
    for ext in ('png','pdf'):
        path=out/f'benchmark_paired_differences.{ext}';fig.savefig(path,dpi=180);record[ext]=str(path.resolve())
    plt.close(fig);return record
