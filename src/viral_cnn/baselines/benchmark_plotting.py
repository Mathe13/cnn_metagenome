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
