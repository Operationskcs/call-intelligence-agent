# Call Intelligence Agent

## Webhook Architecture

```text
PhoneBurner call ends
  -> POST /webhook/phoneburner
  -> X-Webhook-Token validation
  -> Intake CallEvent
  -> existing pipeline s2-s8
  -> call_audit_log
  -> Intake CRM note / manual review

RingCentral call recording completes
  -> POST /webhook/ringcentral
  -> Validation-Token echo during subscription setup
  -> X-Webhook-Token validation for notifications
  -> MedHub CallEvent
  -> BackgroundTasks pipeline s2-s8
  -> call_audit_log
  -> MedHub CRM note / manual review

Manual Intake reprocessing
  -> POST /reprocess/intake
  -> legacy PhoneBurner BigQuery polling path
  -> existing pipeline s2-s8
```

The Cloud Run Job and Cloud Scheduler polling path remain in place while the
webhook service is validated. New real-time processing runs as a Cloud Run
Service with `min-instances=1` and `max-instances=10`.

## Endpoints

- `GET /health` returns `{"status": "ok"}`.
- `POST /webhook/phoneburner` receives PhoneBurner Intake call completion events.
- `POST /webhook/ringcentral` receives RingCentral MedHub push notifications.
- `POST /reprocess/intake` runs one legacy PhoneBurner polling cycle for manual reprocessing.

Webhook and reprocess requests must include:

```text
X-Webhook-Token: <WEBHOOK_SECRET_TOKEN>
```

RingCentral validation requests are handled before shared-token validation and
echo the incoming `Validation-Token` header as required by RingCentral.

## RingCentral Setup

Set the public service URL:

```text
RINGCENTRAL_WEBHOOK_URL=https://<cloud-run-service-url>/webhook/ringcentral
```

On service startup, the worker registers a RingCentral webhook subscription for:

```text
/restapi/v1.0/account/~/telephony/sessions
```

The service stores the subscription ID in memory and renews it every 23 hours via
RingCentral `PUT /restapi/v1.0/subscription/{subscriptionId}`.

To deploy the service:

```bash
gcloud secrets create WEBHOOK_SECRET_TOKEN --replication-policy=automatic
printf '%s' '<shared-token>' | gcloud secrets versions add WEBHOOK_SECRET_TOKEN --data-file=-
gcloud builds submit --config cloudbuild.service.yaml \
  --substitutions _RINGCENTRAL_WEBHOOK_URL=https://<cloud-run-service-url>/webhook/ringcentral
```

## PhoneBurner Setup

Configure the PhoneBurner webhook URL to the deployed service endpoint:

```text
https://<cloud-run-service-url>/webhook/phoneburner
```

PhoneBurner requests must include the shared token header:

```text
X-Webhook-Token: <WEBHOOK_SECRET_TOKEN>
```

Eligible Intake calls must be connected, at least 30 seconds long, and include a
`recording_gcs_uri`.

## Environment Variables

- `WEBHOOK_SECRET_TOKEN`: shared secret required in `X-Webhook-Token`.
- `RINGCENTRAL_WEBHOOK_URL`: public RingCentral callback URL.
- `RINGCENTRAL_CLIENT_ID`, `RINGCENTRAL_CLIENT_SECRET`, `RINGCENTRAL_JWT`, `RINGCENTRAL_ACCOUNT_ID`: RingCentral API credentials.
- `DATABASE_URL`: Cloud SQL/PostgreSQL connection for `call_audit_log`.
- `DEEPGRAM_API_KEY`: transcription provider credential.
- `INTAKE_CRM_BASE_URL`, `INTAKE_CRM_API_TOKEN`, `MEDHUB_CRM_BASE_URL`, `MEDHUB_CRM_API_TOKEN`, `GRS_CRM_BASE_URL`, `GRS_CRM_API_TOKEN`: CRM workspace configuration.

## Validation

Before disabling Cloud Scheduler, validate:

```bash
uv run ruff check .
uv run mypy .
uv run pytest
curl -i https://<cloud-run-service-url>/health
```

Then send simulated PhoneBurner and RingCentral payloads with the shared token and
confirm successful `call_audit_log` rows before disabling the scheduler.
