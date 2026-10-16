"""M19-E4B: single isolated target-only control for finalized E4."""
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e4b_target_only as e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--e4', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    for command in ('verify', 'inspect', 'preview', 'next', 'recover', 'finalize'):
        p = sub.add_parser(command)
        p.add_argument('--out', required=True, type=Path)
        if command == 'next':
            p.add_argument('--allow-paid', action='store_true')
            p.add_argument('--max-new-paid-attempts', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = e.prepare(e4_out=args.e4, out=args.out)
    elif args.command == 'next':
        result = e.next_step(args.out, allow_paid=args.allow_paid,
                             max_new_paid_attempts=args.max_new_paid_attempts)
    else:
        result = getattr(e, args.command)(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
