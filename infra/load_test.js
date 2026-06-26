/**
 * Adaptive RAG Engine — k6 Load Test Scenarios
 *
 * Usage:
 *   k6 run infra/load_test.js                          # smoke test
 *   k6 run -e SCENARIO=load infra/load_test.js         # load test (50 VUs)
 *   k6 run -e SCENARIO=stress infra/load_test.js       # stress test (200 VUs)
 *   k6 run -e SCENARIO=soak infra/load_test.js         # soak test (30min)
 *
 * Requires: k6 (https://k6.io/docs/get-started/installation/)
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// ── Custom metrics ──────────────────────────────────────────────────────────

const queryDuration = new Trend('rag_query_duration', true);
const queryErrors = new Counter('rag_query_errors');
const querySuccessRate = new Rate('rag_query_success_rate');

// ── Configuration ───────────────────────────────────────────────────────────

const BASE_URL = __ENV.BASE_URL || 'http://localhost:80';
const JWT_TOKEN = __ENV.JWT_TOKEN || '';
const SCENARIO = __ENV.SCENARIO || 'smoke';

const scenarios = {
    smoke: {
        executor: 'constant-vus',
        vus: 1,
        duration: '30s',
    },
    load: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            { duration: '1m', target: 10 },    // ramp up
            { duration: '3m', target: 50 },    // sustained load
            { duration: '1m', target: 0 },     // ramp down
        ],
    },
    stress: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            { duration: '1m', target: 50 },
            { duration: '2m', target: 100 },
            { duration: '2m', target: 200 },   // peak stress
            { duration: '1m', target: 0 },
        ],
    },
    soak: {
        executor: 'constant-vus',
        vus: 20,
        duration: '30m',
    },
};

export const options = {
    scenarios: {
        default: scenarios[SCENARIO] || scenarios.smoke,
    },
    thresholds: {
        http_req_duration: ['p(95)<2000', 'p(99)<5000'],   // p95 < 2s, p99 < 5s
        http_req_failed: ['rate<0.01'],                      // <1% error rate
        rag_query_success_rate: ['rate>0.95'],               // >95% query success
        rag_query_duration: ['p(95)<3000'],                  // p95 query < 3s
    },
};

// ── Auth helper ─────────────────────────────────────────────────────────────

function getHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (JWT_TOKEN) {
        headers['Authorization'] = `Bearer ${JWT_TOKEN}`;
    }
    return headers;
}

// ── Test queries ────────────────────────────────────────────────────────────

const testQueries = [
    { query: 'What is retrieval-augmented generation?', strategy: 'auto' },
    { query: 'How does the parent-child chunking strategy work?', strategy: 'auto' },
    { query: 'BGE-M3 embedding dimension', strategy: 'auto' },
    { query: '"reciprocal rank fusion" k=60', strategy: 'auto' },
    { query: 'Explain the four-level LLM fallback chain', strategy: 'dense' },
    { query: 'RAGAS faithfulness threshold configuration', strategy: 'sparse' },
    { query: 'Compare dense and sparse retrieval approaches', strategy: 'hybrid' },
    { query: 'How does semantic caching prevent stampedes?', strategy: 'auto' },
    { query: 'OpenSearch BM25 analyzer configuration', strategy: 'sparse' },
    { query: 'What happens when retrieval returns zero results?', strategy: 'auto' },
];

// ── Main test function ──────────────────────────────────────────────────────

export default function () {
    const headers = getHeaders();

    // ── Health check ────────────────────────────────────
    group('Health Check', () => {
        const res = http.get(`${BASE_URL}/api/health`, { timeout: '10s' });
        check(res, {
            'health status 200': (r) => r.status === 200,
            'health status ok or degraded': (r) => {
                try {
                    const body = JSON.parse(r.body);
                    return body.status === 'ok' || body.status === 'degraded';
                } catch {
                    return false;
                }
            },
        });
    });

    // ── Root endpoint ───────────────────────────────────
    group('Root Endpoint', () => {
        const res = http.get(`${BASE_URL}/`, { timeout: '5s' });
        check(res, {
            'root status 200': (r) => r.status === 200,
            'root has name': (r) => {
                try {
                    return JSON.parse(r.body).name.includes('Adaptive RAG');
                } catch {
                    return false;
                }
            },
        });
    });

    // ── Metrics endpoint ────────────────────────────────
    group('Metrics Endpoint', () => {
        const res = http.get(`${BASE_URL}/metrics`, { timeout: '5s' });
        check(res, {
            'metrics status 200': (r) => r.status === 200,
            'metrics has rag counters': (r) => r.body.includes('rag_query_total'),
        });
    });

    // ── Query endpoint (requires auth) ──────────────────
    if (JWT_TOKEN) {
        group('Query Endpoint', () => {
            const q = testQueries[Math.floor(Math.random() * testQueries.length)];
            const payload = JSON.stringify({
                query: q.query,
                strategy: q.strategy,
                top_k: 5,
                use_cache: true,
                use_reranker: true,
                use_compression: true,
            });

            const start = Date.now();
            const res = http.post(`${BASE_URL}/api/query`, payload, {
                headers: headers,
                timeout: '30s',
            });
            const duration = Date.now() - start;

            queryDuration.add(duration);

            const success = check(res, {
                'query status 200': (r) => r.status === 200,
                'query has answer': (r) => {
                    try {
                        return JSON.parse(r.body).answer.length > 0;
                    } catch {
                        return false;
                    }
                },
                'query has sources': (r) => {
                    try {
                        return JSON.parse(r.body).sources.length > 0;
                    } catch {
                        return false;
                    }
                },
                'query has trace_id': (r) => {
                    try {
                        return JSON.parse(r.body).trace_id.length > 0;
                    } catch {
                        return false;
                    }
                },
            });

            querySuccessRate.add(success);
            if (!success) {
                queryErrors.add(1);
            }
        });
    }

    // ── Auth rejection test ─────────────────────────────
    group('Auth Rejection', () => {
        const res = http.post(
            `${BASE_URL}/api/query`,
            JSON.stringify({ query: 'test unauthorized' }),
            { headers: { 'Content-Type': 'application/json' }, timeout: '10s' }
        );
        check(res, {
            'unauth returns 401': (r) => r.status === 401,
        });
    });

    sleep(1);
}

// ── Teardown ────────────────────────────────────────────────────────────────

export function handleSummary(data) {
    const summary = {
        timestamp: new Date().toISOString(),
        scenario: SCENARIO,
        metrics: {
            http_reqs: data.metrics.http_reqs?.values?.count || 0,
            http_req_duration_p95: data.metrics.http_req_duration?.values?.['p(95)'] || 0,
            http_req_duration_p99: data.metrics.http_req_duration?.values?.['p(99)'] || 0,
            http_req_failed_rate: data.metrics.http_req_failed?.values?.rate || 0,
            rag_query_duration_p95: data.metrics.rag_query_duration?.values?.['p(95)'] || 0,
            rag_query_success_rate: data.metrics.rag_query_success_rate?.values?.rate || 0,
        },
    };

    return {
        stdout: JSON.stringify(summary, null, 2) + '\n',
    };
}
