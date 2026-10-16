"""Inspect and safely recover a saved repeated literature-read request without a paid retry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import open_repeated_tool_literature_recovery as compat
from pipeline_core.discovery.scientific_research_control import open_literature_research as work

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect','recover','verify','preview','next','finish'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=work.ARM_NAMES, default='literature_enabled')
    parser.add_argument('--allow-paid', action='store_true')
    args = parser.parse_args()
    if args.command == 'inspect': result = compat.inspect(args.out,args.arm)
    elif args.command == 'recover': result = compat.recover(args.out,args.arm)
    elif args.command == 'verify': result = compat.verify(args.out)
    elif args.command == 'preview': result = compat.preview(args.out,args.arm)
    elif args.command == 'next': result = compat.next_step(args.out,args.arm,allow_paid=args.allow_paid)
    else: result = compat.finish(args.out,args.arm)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__': main()
