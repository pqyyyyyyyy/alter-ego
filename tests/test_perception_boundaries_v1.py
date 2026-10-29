import numpy as np
import pytest
from alter_ego.perception.identity import FileIdentityStore
from alter_ego.perception.pipeline import PerceptionPipeline,PerceptionConfig
from test_perception_pipeline import FakeBackend,FakeStore,FRAME,box

@pytest.mark.parametrize('vec',[np.zeros(512),np.ones(511),np.full(512,np.nan),np.ones((1,512))])
def test_invalid_vector_rejected(tmp_path,vec):
    store=FileIdentityStore(tmp_path)
    with pytest.raises(ValueError):store.enroll('虚构',[vec]*3)
    assert not store.list_users()

def test_enroll_and_delete_failure_restore_memory(tmp_path,monkeypatch):
    store=FileIdentityStore(tmp_path);v=np.zeros(512);v[0]=1
    original=store._save_index
    def fail():raise PermissionError('fixture write denied')
    monkeypatch.setattr(store,'_save_index',fail)
    with pytest.raises(PermissionError):store.enroll('虚构',[v]*3)
    assert store.list_users()==[];assert list(tmp_path.glob('*.npz'))==[]
    monkeypatch.setattr(store,'_save_index',original);uid=store.enroll('虚构',[v]*3)
    monkeypatch.setattr(store,'_save_index',fail)
    with pytest.raises(PermissionError):store.delete(uid)
    assert store.get(uid);assert store._vec_path(uid).is_file()

@pytest.mark.parametrize('vec',[np.zeros(512),np.ones(511),np.full(512,np.inf)])
def test_bad_backend_vector_does_not_vote(vec):
    b=FakeBackend([1]*6);b.extract_embedding=lambda *args:vec
    store=FakeStore();p=PerceptionPipeline(b,store,PerceptionConfig(vote_max_frames=2))
    for _ in range(5):p.step(FRAME)
    assert p.snapshot.primary.user_id is None;assert store.i==0

def test_missing_keypoints_visitor_and_leave_clear():
    b=FakeBackend([1]*5);face=box();face.kps=None;b.detect_faces=lambda frame:[face]
    p=PerceptionPipeline(b,FakeStore(),PerceptionConfig(vote_max_frames=2))
    for _ in range(5):p.step(FRAME)
    assert p.snapshot.primary.user_id is None;assert b.embed_calls==0
    for _ in range(15):p.step(None)
    assert p.snapshot.primary is None


def test_multi_face_is_anonymous_no_speaker_guess():
    b=FakeBackend([1,1,2,2,1]);store=FakeStore()
    p=PerceptionPipeline(b,store,PerceptionConfig(enter_frames=1,vote_samples=1))
    p.step(FRAME);assert p.snapshot.primary.user_id=='u1'
    p.step(FRAME);before=b.embed_calls
    p.step(FRAME);p.step(FRAME)
    assert p.snapshot.primary.user_id is None;assert b.embed_calls==before
    p.step(FRAME);assert p.snapshot.primary.user_id=='u1'
