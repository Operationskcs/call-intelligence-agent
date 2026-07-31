"""Application settings loaded from environment variables.

Production deployments should inject secrets through GCP Secret Manager. Local
development may use a `.env` file that follows `.env.example`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Base class shared by nested settings models."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


class GCPSettings(EnvSettings):
    """GCP project, region, and Pub/Sub resource settings."""

    project_id: str = Field(default="", validation_alias="GCP_PROJECT_ID")
    region: str = Field(default="", validation_alias="GCP_REGION")
    vertex_region: str = Field(default="us-east5", validation_alias="VERTEX_REGION")
    bq_project: str = Field(default="keep-calm-database", validation_alias="BQ_PROJECT")
    pubsub_topic_call_completed: str = Field(
        default="call.completed",
        validation_alias="PUBSUB_TOPIC_CALL_COMPLETED",
    )
    pubsub_subscription_worker: str = Field(
        default="call-intelligence-worker-sub",
        validation_alias="PUBSUB_SUBSCRIPTION_WORKER",
    )


class GCSSettings(EnvSettings):
    """GCS buckets used by telephony providers."""

    audio_bucket: str = Field(
        default="pb-dispositions-call-recordings",
        validation_alias="GCS_AUDIO_BUCKET",
    )
    ringcentral_bucket: str = Field(
        default="medhub-call-recordings",
        validation_alias="GCS_RINGCENTRAL_BUCKET",
    )


class PhoneBurnerSettings(EnvSettings):
    """PhoneBurner API and webhook settings."""

    api_key: str = Field(default="", validation_alias="PHONEBURNER_API_KEY")
    client_id: str = Field(default="", validation_alias="PHONEBURNER_CLIENT_ID")
    webhook_secret: str = Field(default="", validation_alias="PHONEBURNER_WEBHOOK_SECRET")


class RingCentralSettings(EnvSettings):
    """RingCentral API and webhook settings."""

    client_id: str = Field(default="", validation_alias="RINGCENTRAL_CLIENT_ID")
    client_secret: str = Field(default="", validation_alias="RINGCENTRAL_CLIENT_SECRET")
    jwt: str = Field(default="", validation_alias="RINGCENTRAL_JWT")
    account_id: str = Field(default="", validation_alias="RINGCENTRAL_ACCOUNT_ID")
    webhook_url: str = Field(default="", validation_alias="RINGCENTRAL_WEBHOOK_URL")
    webhook_verification_token: str = Field(
        default="",
        validation_alias="RINGCENTRAL_WEBHOOK_VERIFICATION_TOKEN",
    )


class TelephonySettings(EnvSettings):
    """Telephony provider settings grouped by source."""

    phoneburner: PhoneBurnerSettings = Field(default_factory=PhoneBurnerSettings)
    ringcentral: RingCentralSettings = Field(default_factory=RingCentralSettings)


class STTSettings(EnvSettings):
    """Speech-to-text provider settings."""

    deepgram_api_key: str = Field(default="", validation_alias="DEEPGRAM_API_KEY")
    provider: Literal["deepgram", "google", "passthrough"] = Field(
        default="deepgram",
        validation_alias="STT_PROVIDER",
    )


class WorkspaceCRMSettings(EnvSettings):
    """Twenty CRM connection settings for one workspace."""

    base_url: str = ""
    api_token: str = ""


class CRMSettings(EnvSettings):
    """Twenty CRM workspace settings."""

    intake: WorkspaceCRMSettings = Field(default_factory=WorkspaceCRMSettings)
    medhub: WorkspaceCRMSettings = Field(default_factory=WorkspaceCRMSettings)
    grs: WorkspaceCRMSettings = Field(default_factory=WorkspaceCRMSettings)

    @classmethod
    def from_env(cls) -> "CRMSettings":
        """Build CRM settings from flat workspace-specific environment variables."""

        return cls(
            intake=WorkspaceCRMSettings(
                base_url=IntakeCRMEnv().base_url,
                api_token=IntakeCRMEnv().api_token,
            ),
            medhub=WorkspaceCRMSettings(
                base_url=MedHubCRMEnv().base_url,
                api_token=MedHubCRMEnv().api_token,
            ),
            grs=WorkspaceCRMSettings(
                base_url=GRSCRMEnv().base_url,
                api_token=GRSCRMEnv().api_token,
            ),
        )


class IntakeCRMEnv(EnvSettings):
    """Flat environment aliases for Intake CRM settings."""

    base_url: str = Field(default="", validation_alias="INTAKE_CRM_BASE_URL")
    api_token: str = Field(default="", validation_alias="INTAKE_CRM_API_TOKEN")


class MedHubCRMEnv(EnvSettings):
    """Flat environment aliases for MedHub CRM settings."""

    base_url: str = Field(default="", validation_alias="MEDHUB_CRM_BASE_URL")
    api_token: str = Field(default="", validation_alias="MEDHUB_CRM_API_TOKEN")


class GRSCRMEnv(EnvSettings):
    """Flat environment aliases for GRS CRM settings."""

    base_url: str = Field(default="", validation_alias="GRS_CRM_BASE_URL")
    api_token: str = Field(default="", validation_alias="GRS_CRM_API_TOKEN")


class DatabaseSettings(EnvSettings):
    """Database connection settings."""

    url: str = Field(default="", validation_alias="DATABASE_URL")


class PipelineSettings(EnvSettings):
    """Pipeline behavior and review queue settings."""

    call_quality_trigger_url: str = Field(
        default=(
            "https://call-quality-trigger-znesczxkka-uc.a.run.app/"
            "integrations/call-intelligence/trigger"
        ),
        validation_alias="CALL_QUALITY_TRIGGER_URL",
    )
    min_call_duration_seconds: int = Field(
        default=30,
        validation_alias="MIN_CALL_DURATION_SECONDS",
    )
    match_confidence_threshold: float = Field(
        default=0.8,
        validation_alias="MATCH_CONFIDENCE_THRESHOLD",
    )
    poll_interval_minutes: int = Field(default=5, validation_alias="POLL_INTERVAL_MINUTES")
    phoneburner_lookback_hours: int = Field(
        default=24,
        validation_alias="PHONEBURNER_LOOKBACK_HOURS",
    )
    ringcentral_lookback_minutes: int = Field(
        default=5,
        validation_alias="RINGCENTRAL_LOOKBACK_MINUTES",
    )
    slack_review_queue_webhook_url: str = Field(
        default="",
        validation_alias="SLACK_REVIEW_QUEUE_WEBHOOK_URL",
    )


class WebhookSettings(EnvSettings):
    """Shared webhook receiver settings."""

    secret_token: str = Field(default="", validation_alias="WEBHOOK_SECRET_TOKEN")


class Settings(EnvSettings):
    """Root settings object composed of nested settings groups."""

    gcp: GCPSettings = Field(default_factory=GCPSettings)
    gcs: GCSSettings = Field(default_factory=GCSSettings)
    telephony: TelephonySettings = Field(default_factory=TelephonySettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    crm: CRMSettings = Field(default_factory=CRMSettings.from_env)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)

    def safe_summary(self) -> dict[str, object]:
        """Return non-secret settings useful for startup logs."""

        return {
            "gcp_project_id": self.gcp.project_id,
            "gcp_region": self.gcp.region,
            "vertex_region": self.gcp.vertex_region,
            "bq_project": self.gcp.bq_project,
            "poll_interval_minutes": self.pipeline.poll_interval_minutes,
            "call_quality_trigger_url": self.pipeline.call_quality_trigger_url,
            "phoneburner_lookback_hours": self.pipeline.phoneburner_lookback_hours,
            "ringcentral_lookback_minutes": self.pipeline.ringcentral_lookback_minutes,
            "pubsub_topic": self.gcp.pubsub_topic_call_completed,
            "pubsub_subscription": self.gcp.pubsub_subscription_worker,
            "audio_bucket": self.gcs.audio_bucket,
            "ringcentral_bucket_configured": bool(self.gcs.ringcentral_bucket),
            "webhook_secret_configured": bool(self.webhook.secret_token),
            "ringcentral_webhook_url_configured": bool(
                self.telephony.ringcentral.webhook_url
            ),
            "stt_provider": self.stt.provider,
            "database_configured": bool(self.database.url),
            "review_queue_configured": bool(self.pipeline.slack_review_queue_webhook_url),
            "crm_workspaces_configured": {
                "intake": bool(self.crm.intake.base_url),
                "medhub": bool(self.crm.medhub.base_url),
                "grs": bool(self.crm.grs.base_url),
            },
        }


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
