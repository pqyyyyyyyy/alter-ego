"""Similarity ROC counts, not an estimate from synthetic face accuracy."""
import csv,math

def evaluate(rows, thresholds=None):
    rows=list(rows);thresholds=list(thresholds) if thresholds is not None else [i/100 for i in range(0,101)]
    seen=set();positive=[];negative=[]
    for r in rows:
        if r['pair_id'] in seen:raise ValueError('duplicate pair_id: do not inflate denominators')
        seen.add(r['pair_id'])
        label=r['label']
        if label not in ('match','nonmatch'):raise ValueError('label must be match/nonmatch')
        x=float(r['similarity'])
        if not math.isfinite(x) or not -1<=x<=1:raise ValueError('invalid cosine similarity')
        (positive if label=='match' else negative).append(x)
    if not positive or not negative:raise ValueError('both genuine and impostor pairs required')
    output=[]
    for threshold in thresholds:
        if not math.isfinite(threshold) or not -1<=threshold<=1:raise ValueError('invalid threshold')
        fa=sum(x>=threshold for x in negative);fr=sum(x<threshold for x in positive)
        output.append(dict(threshold=threshold,far=fa/len(negative),frr=fr/len(positive),false_accepts=fa,false_rejects=fr,n_positive=len(positive),n_negative=len(negative)))
    return output

def read_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
