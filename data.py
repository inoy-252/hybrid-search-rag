"""
Internal Engineering Knowledge Base: Technical Runbooks & System Specs.
Used for evaluating Hybrid Search (Lexical BM25 + Dense Semantic Vectors).
"""

DOCUMENTS = [
    {
        "id": "DOC-AUTH-101",
        "title": "Authentication Service & Token Rotation Runbook",
        "category": "Security & Identity",
        "content": (
            "The Authentication Service manages user sessions, OAuth2 handshakes, and JWT issuance. "
            "Tokens are signed with RSA-256 and expire every 15 minutes. In the event of compromised credentials, "
            "engineers must trigger manual token invalidation using the CLI command 'auth-cli revoke-all --tenant-id'. "
            "If clients receive error code ERR-AUTH-401-EXPIRED, verify that the system clock drift across worker nodes "
            "does not exceed 500 milliseconds. Secret rotation keys are stored in Vault under path /secret/auth/master-key."
        ),
    },
    {
        "id": "DOC-DB-202",
        "title": "PostgreSQL Primary Cluster Failover & Replication",
        "category": "Database Infrastructure",
        "content": (
            "The primary transactional database runs on PostgreSQL 16 on instance DB-CLUSTER-PRIMARY. "
            "Streaming replication writes to read-replicas DB-REPLICA-EAST and DB-REPLICA-WEST. "
            "When the primary node fails health checks for 30 consecutive seconds, Patroni automatically triggers "
            "leader election and promotes DB-REPLICA-EAST to leader. For disaster recovery and manual failovers, "
            "run 'patronictl failover cluster-prod' and verify replication lag is zero using query 'SELECT pg_stat_replication'. "
            "Critical connection pooler PgBouncer listens on internal port 6432."
        ),
    },
    {
        "id": "DOC-INFRA-303",
        "title": "High Traffic Load Shedding and Auto-Scaling Guide",
        "category": "Site Reliability & Cloud",
        "content": (
            "During traffic surges and flash-sale events, the infrastructure relies on Horizontal Pod Autoscaling (HPA). "
            "When CPU utilization across pods exceeds 75% for 2 minutes, Kubernetes automatically spins up replica pods up to 50 nodes. "
            "If downstream backend services start suffocating, the API Gateway activates adaptive load shedding, returning HTTP 429 "
            "Too Many Requests to non-critical background traffic. To handle unexpected server crashes, Envoy rate-limiters throttle incoming "
            "connections using a token bucket algorithm to ensure core checkout operations remain 100% operational."
        ),
    },
    {
        "id": "DOC-PAY-404",
        "title": "Payment Gateway Reconciliation & Error Code Reference",
        "category": "Financial Services",
        "content": (
            "The Payments Engine processes credit cards, UPI, and bank transfers via Stripe and Razorpay integrations. "
            "All payment transactions require an idempotency key passed in header 'X-Idempotency-Key' to avoid duplicate charges. "
            "If webhook callbacks fail, the system retries with exponential backoff up to 7 attempts. "
            "Frequent error codes: ERR-PAY-502-GATEWAY indicates merchant banking gateway timeout; "
            "ERR-PAY-409-DUPLICATE indicates an idempotency collision; ERR-PAY-402-DECLINED represents card rejection. "
            "Daily automated settlement reconciliation runs at 02:00 UTC."
        ),
    },
    {
        "id": "DOC-OBS-505",
        "title": "Distributed Tracing and Observability with OpenTelemetry",
        "category": "Observability & Monitoring",
        "content": (
            "All production microservices emit metrics, logs, and distributed traces via OpenTelemetry collector agents. "
            "Traces are collected and indexed in Grafana Tempo, while Prometheus scrapes metrics on port 9090. "
            "Every HTTP request injects a correlation header 'X-Trace-ID' to track call graphs across microservice hops. "
            "When p99 latency exceeds 2.5 seconds, PagerDuty automatically alerts on-call engineers. "
            "To debug silent memory leaks or hanging threads, dump heap metrics using 'kill -SIGUSR1 <pid>' to generate memory snapshots."
        ),
    },
]
