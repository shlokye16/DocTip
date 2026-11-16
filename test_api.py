"""FastAPI smoke tests; no API key or external network required."""
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_hosted_frontend():
    for path, content_type in [('/', 'text/html'), ('/index.html', 'text/html'),
                               ('/evals.html', 'text/html'), ('/blob.png', 'image/png')]:
        response = client.get(path)
        assert response.status_code == 200
        assert content_type in response.headers['content-type']
    assert client.get('/server.py').status_code == 404
    assert client.get('/.env').status_code == 404


def test_health_and_profiles():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['profiles_loaded'] == 7


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


def test_added_conditions_api_and_cds_cards():
    for text, pid, code in [('Endometriosis', 'endometriosis', 'N80.9'),
                            ('HIV', 'hiv', 'B20'),
                            ('Depression', 'depression', 'F32.A'),
                            ('Tuberculosis', 'tuberculosis', 'A15.0'),
                            ('Heart disease', 'acs_chest_pain', 'I25.10')]:
        sid = client.post('/api/sessions').json()['session_id']
        response = client.post(f'/api/sessions/{sid}/turn', json={'text': text})
        assert response.status_code == 200 and response.json()['disease']['id'] == pid
        assert client.post(f'/api/sessions/{sid}/end').json()['phase'] == 'debrief'
        assert client.delete(f'/api/sessions/{sid}').status_code == 200
        payload = {'hook': 'patient-view', 'context': {'patientId': 'synthetic'},
                   'prefetch': {'conditions': {'entry': [{'resource': {'code': {
                       'coding': [{'code': code}]}}}]}}}
        cards = client.post('/cds-services/clinician-reasoning-check', json=payload).json()['cards']
        assert len(cards) == 1 and cards[0]['indicator'] == 'info'


def test_session_flags_only_configured_conditions():
    created = client.post('/api/sessions')
    sid = created.json()['session_id']
    quiet = client.post(f'/api/sessions/{sid}/turn', json={'text': 'Patient has a mild ankle sprain.'})
    assert quiet.status_code == 200 and quiet.json()['flagged'] is False
    flagged = client.post(f'/api/sessions/{sid}/turn', json={'text': 'The patient has sickle cell disease.'})
    assert flagged.status_code == 200 and flagged.json()['flagged'] is True
    assert flagged.json()['disease']['id'] == 'sickle_cell'
    assert flagged.json()['phase'] == 'visit' and flagged.json()['reply'] is None
    ended = client.post(f'/api/sessions/{sid}/end')
    assert ended.status_code == 200 and ended.json()['phase'] == 'debrief' and '?' in ended.json()['reply']
    assert client.post('/api/sessions/nope/end').status_code == 404
    deleted = client.delete(f'/api/sessions/{sid}')
    assert deleted.status_code == 200


def test_quick_reflection_records_selected_answers_once():
    for condition in ['endometriosis', 'sickle cell', 'depression', 'heart disease',
                      'tuberculosis', 'HIV', 'preeclampsia']:
        sid = client.post('/api/sessions').json()['session_id']
        path = f'/api/sessions/{sid}'
        client.post(path + '/turn', json={'text': condition})
        questions = client.post(path + '/end?quick=true').json()['reflection']
        assert len(questions) == 2
        assert client.post(path + '/reflection', json={'answers': []}).status_code == 422
        answers = [questions[0]['options'][0], 'Unsure']
        for _ in range(2):
            saved = client.post(path + '/reflection', json={'answers': answers})
            assert saved.status_code == 200
        note = client.get(path + '/note').json()
        assert [a['answer'] for a in note['json']['qa']] == answers
        assert 'answer judged' not in note['text']
        client.delete(path)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    print('ALL API SMOKE TESTS PASSED')
