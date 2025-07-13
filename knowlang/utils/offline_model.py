from pydantic_ai.models import Model
from pydantic_ai.messages import ModelResponse, TextPart


class OfflineModel(Model):
    async def request(
        self,
        messages,
        model_settings,
        model_request_parameters,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content="This is an offline model. No actual request will be made."
                )
            ],
            model_name=self.model_name,
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return self.__class__.__name__

    @property
    def system(self) -> str:
        """The system / model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        return self.__class__.__name__

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None
