"""Data layer and ingestion clients for Lekka AI."""

from lekka_ai.data.cache import Cache
from lekka_ai.data.filings import (
    clean_html_to_text,
    get_section,
    parse_filing_document,
    split_filing_sections,
)
from lekka_ai.data.http import (
    AsyncHttpClient,
    HostRateLimiter,
    RetryableHTTPStatusError,
    TokenBucket,
    create_http_client,
)
from lekka_ai.data.market import (
    CsvFixtureProvider,
    MarketDataError,
    MarketDataProvider,
    PaidMarketDataProvider,
    TickerNotFoundError,
    YFinanceProvider,
)
from lekka_ai.data.sec import (
    CIKNotFoundError,
    SecClient,
    SecClientError,
)
from lekka_ai.data.xbrl import (
    DEFAULT_TAG_FALLBACKS,
    CompanyFactsParser,
    CompanyFactsResult,
    MetricSource,
    PeriodFinancials,
    parse_company_facts,
)

__all__ = [
    "AsyncHttpClient",
    "CIKNotFoundError",
    "Cache",
    "CompanyFactsParser",
    "CompanyFactsResult",
    "CsvFixtureProvider",
    "DEFAULT_TAG_FALLBACKS",
    "HostRateLimiter",
    "MarketDataError",
    "MarketDataProvider",
    "MetricSource",
    "PaidMarketDataProvider",
    "PeriodFinancials",
    "RetryableHTTPStatusError",
    "SecClient",
    "SecClientError",
    "TickerNotFoundError",
    "TokenBucket",
    "YFinanceProvider",
    "clean_html_to_text",
    "create_http_client",
    "get_section",
    "parse_company_facts",
    "parse_filing_document",
    "split_filing_sections",
]
