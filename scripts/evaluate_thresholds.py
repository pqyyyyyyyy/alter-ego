import argparse,csv,json
from pathlib import Path
from alter_ego.perception.thresholds import read_csv,evaluate
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',required=True);a=p.parse_args()
rows=read_csv(a.csv);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);scan=evaluate(rows)
with (out/'threshold-scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(scan[0]));w.writeheader();w.writerows(scan)
(out/'metadata.json').write_text(json.dumps(dict(n_pairs=len(rows),conditions=sorted({(r.get('distance'),r.get('lighting'),r.get('split')) for r in rows}),scope='similarity counts only; no auto selection; threshold 0.40 unchanged'),ensure_ascii=False,indent=2),encoding='utf8')
print('No model/camera was used; scan saved',out)
