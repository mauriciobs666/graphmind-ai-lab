"""Settings read from the application config file."""
from pydantic import AliasPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config/app.yaml")

    db_pool_size: int = Field(5, validation_alias=AliasPath("db", "pool_size"))


settings = Settings()
