"""FastAPI smoke tests; no API key or external network required."""
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_health_and_profiles():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['profiles_loaded'] == 3


def test_discovery_and_cds_card():
    discovery = client.get('/cds-services')
    assert discovery.status_code == 200
    assert discovery.json()['services'][0]['hook'] == 'patient-view'
    payload = {
        'hook': 'patient-view', 'hookInstance': 'test-hook',
        'context': {'patientId': 'synthetic-patient'},
        'prefetch': {'conditions': {'entry': [{'resource': {'code': {
            'coding': [{'code': 'D57.00'}]
        }}}]}}
    }
    result = client.post('/cds-services/clinician-reasoning-check', json=payload)
    assert result.status_code == 200
    assert len(result.json()['cards']) == 1
    assert 'Sickle cell' in result.json()['cards'][0]['summary']


def test_session_flags_only_configured_conditions():
    created = client.post('/api/sessions')
    sid = created.json()['session_id']
    quiet = client.post(f'/api/sessions/{sid}/turn', json={'text': 'Patient has a mild ankle sprain.'})
    assert quiet.status_code == 200 and quiet.json()['flagged'] is False
    flagged = client.post(f'/api/sessions/{sid}/turn', json={'text': 'The patient has sickle cell disease.'})
    assert flagged.status_code == 200 and flagged.json()['flagged'] is True
    assert flagged.json()['disease']['id'] == 'sickle_cell'
    assert flagged.json()['phase'] == 'visit' and flagged.json()['reply'].endswith('.')
    ended = client.post(f'/api/sessions/{sid}/end')
    assert ended.status_code == 200 and ended.json()['phase'] == 'debrief' and '?' in ended.json()['reply']
    assert client.post('/api/sessions/nope/end').status_code == 404
    deleted = client.delete(f'/api/sessions/{sid}')
    assert deleted.status_code == 200


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    print('ALL API SMOKE TESTS PASSED')
