| # | item | anchor | impacted element | mechanism | now | status/distance or reason |
|---|---|---|---|---|---|---|
| 0 | 021708 | GET /health | coordination-techniques.md::<document> | curl URL in inline Markdown span (carrier) | miss | answer lists 10 items; the Markdown `<document>` is not among them (its `curl` call site has no valueSet) |
| 1 | 0414ef | GET /workflow-defs | app.js::loadDefsList | already built (hit) | hit | possible, distance 1 |
| 2 | 0a94dc | GET /health | load_append.py::main | already built (hit) | hit | confirmed, distance 1 |
| 3 | 1f4925 | GET /workspaces/{}/snapshots/{}/versions/{} | app.js::renderWaitingForm | already built (hit) | hit | possible, distance 1 |
| 4 | 40c8c6 | POST /shop/api/session | endpoints.ts::joinSession | already built (hit) | hit | confirmed, distance 1 |
| 5 | 47f126 | POST /threads/{}/messages | app.js::postMessage | already built (hit) | hit | possible, distance 1 |
| 6 | 57f718 | GET /health | seed_nlq_eval_corpus.py::_check_server_reachable | already built (hit) | hit | confirmed, distance 1 |
| 7 | 7546b0 | GET /channels | app.js::loadChannels | already built (hit) | hit | possible, distance 1 |
| 8 | 7f3682 | POST /threads/{}/messages | load_append.py::_worker | already built (hit) | hit | possible, distance 1 |
| 9 | 8ffa9f | POST /workflow-runs/{}/input | app.js::submitRunInput | already built (hit) | hit | possible, distance 1 |
| 10 | 9703a1 | GET /documents/{} | seed_nlq_eval_corpus.py::_poll_until_terminal | already built (hit) | hit | possible, distance 1 |
| 11 | c05679 | POST /shop/api/session | language_salience_probe.py::run_participant | already built (hit) | hit | possible, distance 1 |
| 12 | c58283 | GET /channels/{}/threads | app.js::loadThreads | already built (hit) | hit | possible, distance 1 |
| 13 | e25af0 | GET /shop/api/health | endpoints.ts::getHealth | already built (hit) | hit | confirmed, distance 1 |
| 14 | f2d84c | GET /shop/api/messages | endpoints.ts::getMessages | already built (hit) | hit | confirmed, distance 1 |
| 15 | f3f1d5 | GET /shop/api/messages | language_salience_probe.py::_assistant_rows_since | already built (hit) | hit | possible, distance 1 |
| 16 | f65a45 | POST /documents | seed_nlq_eval_corpus.py::main | already built (hit) | hit | confirmed, distance 1 |
| 17 | f9041b | POST /shop/api/session | load_demo.py::run_participant | already built (hit) | hit | possible, distance 1 |
