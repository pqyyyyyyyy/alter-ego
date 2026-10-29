import pytest
from alter_ego.perception.thresholds import evaluate

def row(i,x,label):return dict(pair_id=str(i),similarity=x,label=label)
def test_denominators_and_boundary():
    out=evaluate([row(1,.9,'match'),row(2,.5,'match'),row(3,.8,'nonmatch'),row(4,.1,'nonmatch')],[.5,.8,.9])
    assert out[0]['n_positive']==2 and out[0]['n_negative']==2
    assert out[0]['frr']==0 and out[0]['far']==.5
    assert out[1]['far']==.5 and out[1]['frr']==.5
    assert out[2]['far']==0
@pytest.mark.parametrize('rows',[[row(1,.9,'match')],[row(1,.9,'unknown')],[row(1,float('nan'),'match'),row(2,.1,'nonmatch')],[row(1,.9,'match'),row(1,.1,'nonmatch')],[row(1,2,'match'),row(2,.1,'nonmatch')]])
def test_invalid_pairs_rejected(rows):
    with pytest.raises(ValueError):evaluate(rows)
