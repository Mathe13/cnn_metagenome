#!/usr/bin/env python3
"""Reserve and inspect an M00 reproduction round without executing notebooks."""
import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from viral_cnn.run_planning import build_run_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--project-root', default=str(ROOT))
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--run-id')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    try:
        plan = build_run_plan(args.config, args.project_root, data_root=args.data_root,
                              run_id=args.run_id, smoke_test=args.smoke_test)
        print(json.dumps({'status': plan['status'], 'plan_path': plan['plan_path'],
                          'stages': plan['preflight_summary'],
                          'report_path': str(Path(plan['plan_path']).with_name('preflight_report.json'))}, indent=2))
        return 0 if plan['status'] == 'READY' else 2
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc)}, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
