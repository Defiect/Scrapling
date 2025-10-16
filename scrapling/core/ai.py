from asyncio import gather

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from scrapling.core.shell import Convertor
from scrapling.engines.toolbelt.custom import Response as _ScraplingResponse
from scrapling.fetchers import (
    FetcherSession,
    AsyncDynamicSession,
    AsyncStealthySession,
)
from scrapling.core._types import (
    Optional,
    Tuple,
    extraction_types,
    Mapping,
    Dict,
    List,
    Any,
    SelectorWaitStates,
    Generator,
)
from curl_cffi.requests import (
    BrowserTypeLiteral,
)
from typing import Literal

ProtectionLevel = Literal["basic", "playwright", "stealth"]


class ResponseModel(BaseModel):
    """Request's response information structure."""

    status: int = Field(description="The status code returned by the website.")
    content: list[str] = Field(description="The content as Markdown/HTML or the text content of the page.")
    url: str = Field(description="The URL given by the user that resulted in this response.")


def _ContentTranslator(content: Generator[str, None, None], page: _ScraplingResponse) -> ResponseModel:
    """Convert a content generator to a list of ResponseModel objects."""
    return ResponseModel(status=page.status, content=[result for result in content], url=page.url)


class ScraplingMCPServer:
    @staticmethod
    async def fetch_content(
        urls: List[str],
        protection_level: ProtectionLevel = "basic",
        extraction_type: extraction_types = "markdown",
        css_selector: Optional[str] = None,
        main_content_only: bool = True,
        # Basic protection parameters
        impersonate: Optional[BrowserTypeLiteral] = "chrome",
        params: Optional[Dict | List | Tuple] = None,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        cookies: Optional[List[Dict]] = None,
        timeout: int | float = 30,
        follow_redirects: bool = True,
        max_redirects: int = 30,
        retries: Optional[int] = 3,
        retry_delay: Optional[int] = 1,
        proxy: Optional[str | Dict[str, str]] = None,
        proxy_auth: Optional[Tuple[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
        verify: Optional[bool] = True,
        http3: Optional[bool] = False,
        stealthy_headers: Optional[bool] = True,
        # Playwright protection parameters
        headless: bool = False,
        google_search: bool = True,
        hide_canvas: bool = False,
        disable_webgl: bool = False,
        real_chrome: bool = False,
        stealth: bool = False,
        wait: int | float = 0,
        locale: str = "en-US",
        extra_headers: Optional[Dict[str, str]] = None,
        useragent: Optional[str] = None,
        cdp_url: Optional[str] = None,
        disable_resources: bool = False,
        wait_selector: Optional[str] = None,
        network_idle: bool = False,
        wait_selector_state: SelectorWaitStates = "attached",
        # Stealth protection parameters
        block_images: bool = False,
        block_webrtc: bool = False,
        allow_webgl: bool = True,
        humanize: bool | float = True,
        solve_cloudflare: bool = False,
        addons: Optional[List[str]] = None,
        os_randomize: bool = False,
        disable_ads: bool = False,
        geoip: bool = False,
        additional_args: Optional[Dict] = None,
    ) -> List[ResponseModel]:
        """Fetch one or more URLs and return structured content extraction.

        This unified tool handles all web scraping scenarios by adjusting the protection level:
        - "basic": Fast HTTP requests with browser fingerprint impersonation. Best for sites with low-to-mid protection.
        - "playwright": Full browser automation with JavaScript support. Best for sites requiring JS execution or mid-level protection.
        - "stealth": Advanced anti-detection browser (Camoufox) with humanization. Best for high-protection sites (Cloudflare, etc).

        **Examples:**
            # Single URL with default settings
            fetch_content(["https://example.com"])
            
            # Multiple URLs in parallel
            fetch_content(["https://site1.com", "https://site2.com", "https://site3.com"])
            
            # Extract specific content using CSS selector
            fetch_content(["https://example.com"], css_selector="article.main-content")
            
            # Handle JavaScript-heavy site
            fetch_content(["https://spa-site.com"], protection_level="playwright")
            
            # Bypass Cloudflare protection
            fetch_content(["https://protected-site.com"], protection_level="stealth", solve_cloudflare=True)

        **Protection Level Guide:**
            Start with "basic" (fastest) and escalate only if needed:
            - Getting 403/blocked or 429? Try "playwright"
            - Still blocked? Use "stealth"
            - Cloudflare challenge? Use "stealth" with solve_cloudflare=True

        :param urls: List of URLs to fetch. Pass single URL as ["https://example.com"].
        :param protection_level: Security bypass level - "basic" (fastest), "playwright" (JS support), or "stealth" (highest protection). Defaults to "basic".
        :param extraction_type: Content format - "markdown" (default), "html", or "text".
        :param css_selector: Optional CSS selector to extract only specific elements.
        :param main_content_only: If True, extracts only <body> content. Defaults to True.
        :param impersonate: (basic) Browser fingerprint to impersonate; defaults to the latest Chrome profile.
        :param params: (basic) Query string parameters to include with the request.
        :param headers: (basic) Extra headers to include with the request.
        :param cookies: (playwright/stealth only) Cookies in Playwright-compatible dict format for browser modes.
        :param timeout: (basic) Request timeout in seconds. Converted to milliseconds for browser-based modes. Defaults to 30 seconds.
        :param follow_redirects: (basic) Follow HTTP redirects. Defaults to True.
        :param max_redirects: (basic) Maximum number of redirects allowed. Defaults to 30; use -1 for unlimited.
        :param retries: (basic) Number of retry attempts per URL. Defaults to 3.
        :param retry_delay: (basic) Seconds to wait between retries. Defaults to 1.
        :param proxy: (all levels) Proxy configuration as a URL string or dict with Scrapling-supported keys.
        :param proxy_auth: (basic) Optional proxy basic-auth credentials `(username, password)`.
        :param auth: (basic) Optional HTTP basic-auth credentials for the target site.
        :param verify: (basic) Whether to verify TLS certificates. Defaults to True.
        :param http3: (basic) Enable HTTP/3 support. Defaults to False.
        :param stealthy_headers: (basic) Generate real browser headers and Google-style referrer. Defaults to True.
        :param headless: (playwright/stealth) Run the browser in headless mode. Defaults to False for Playwright.
        :param google_search: (playwright/stealth) Spoof Google referrer headers. Defaults to True.
        :param hide_canvas: (playwright) Add noise to Canvas operations to avoid fingerprinting.
        :param disable_webgl: (playwright) Disable WebGL to avoid detection.
        :param real_chrome: (playwright) Attach to the real Chrome browser installed locally.
        :param stealth: (playwright) Enable Playwright stealth mode with anti-detection patches.
        :param wait: (playwright/stealth) Milliseconds to wait after load completes before returning.
        :param locale: (playwright) Browser locale string. Defaults to "en-US".
        :param extra_headers: (playwright/stealth) Additional request headers to send.
        :param useragent: (playwright) Override the automatically generated user agent string.
        :param cdp_url: (playwright) Remote Chrome DevTools endpoint to control instead of launching a browser.
        :param disable_resources: (playwright/stealth) Block non-essential resources (images, fonts, etc.) for faster loads.
        :param wait_selector: (playwright/stealth) CSS selector to wait for before returning.
        :param network_idle: (playwright/stealth) Wait for network to be idle for 500 ms before continuing.
        :param wait_selector_state: (playwright/stealth) Desired state ("attached", "visible", etc.) for `wait_selector`.
        :param block_images: (stealth) Block images at the browser level to save bandwidth.
        :param block_webrtc: (stealth) Disable WebRTC to prevent IP leaks.
        :param allow_webgl: (stealth) Keep WebGL enabled; disable only when required. Defaults to True.
        :param humanize: (stealth) Enable cursor humanization (True) or specify max movement duration.
        :param solve_cloudflare: (stealth) Automatically solve Cloudflare Turnstile/interstitial challenges.
        :param addons: (stealth) List of Firefox add-on paths to load into Camoufox.
        :param os_randomize: (stealth) Randomize OS fingerprints when enabled.
        :param disable_ads: (stealth) Install uBlock Origin to reduce ads.
        :param geoip: (stealth) Derive language/timezone/WebRTC IP from proxy geolocation.
        :param additional_args: (stealth) Extra Camoufox configuration overrides.
        :return: List of ResponseModel objects, one per URL (even if only one URL was provided).
        """
        if not urls:
            return []

        if protection_level not in ("basic", "playwright", "stealth"):
            raise ValueError(
                f"Unsupported protection_level '{protection_level}'. Expected 'basic', 'playwright', or 'stealth'."
            )

        if protection_level == "basic":
            request_kwargs = {
                "auth": auth,
                "http3": http3,
                "verify": verify,
                "params": params,
                "headers": headers,
                "timeout": timeout,
                "retries": retries,
                "proxy_auth": proxy_auth,
                "retry_delay": retry_delay,
                "impersonate": impersonate,
                "max_redirects": max_redirects,
                "follow_redirects": follow_redirects,
                "stealthy_headers": stealthy_headers,
            }
            if isinstance(proxy, dict):
                request_kwargs["proxies"] = proxy
            else:
                request_kwargs["proxy"] = proxy

            async with FetcherSession() as session:
                tasks: List[Any] = [session.get(url, **request_kwargs) for url in urls]
                responses = await gather(*tasks)
        elif protection_level == "playwright":
            playwright_timeout = timeout * 1000
            async with AsyncDynamicSession(
                wait=wait,
                proxy=proxy,
                locale=locale,
                timeout=playwright_timeout,
                cookies=cookies,
                stealth=stealth,
                cdp_url=cdp_url,
                headless=headless,
                max_pages=len(urls),
                useragent=useragent,
                hide_canvas=hide_canvas,
                real_chrome=real_chrome,
                network_idle=network_idle,
                wait_selector=wait_selector,
                google_search=google_search,
                disable_webgl=disable_webgl,
                extra_headers=extra_headers,
                disable_resources=disable_resources,
                wait_selector_state=wait_selector_state,
            ) as session:
                tasks = [session.fetch(url) for url in urls]
                responses = await gather(*tasks)
        else:  # protection_level == "stealth"
            stealth_timeout = timeout * 1000
            async with AsyncStealthySession(
                wait=wait,
                proxy=proxy,
                geoip=geoip,
                addons=addons,
                timeout=stealth_timeout,
                cookies=cookies,
                headless=headless,
                humanize=humanize,
                max_pages=len(urls),
                allow_webgl=allow_webgl,
                disable_ads=disable_ads,
                block_images=block_images,
                block_webrtc=block_webrtc,
                network_idle=network_idle,
                os_randomize=os_randomize,
                wait_selector=wait_selector,
                google_search=google_search,
                extra_headers=extra_headers,
                solve_cloudflare=solve_cloudflare,
                disable_resources=disable_resources,
                wait_selector_state=wait_selector_state,
                additional_args=additional_args,
            ) as session:
                tasks = [session.fetch(url) for url in urls]
                responses = await gather(*tasks)

        return [
            _ContentTranslator(
                Convertor._extract_content(
                    page,
                    css_selector=css_selector,
                    extraction_type=extraction_type,
                    main_content_only=main_content_only,
                ),
                page,
            )
            for page in responses
        ]

    def serve(self, http: bool, host: str, port: int):
        """Serve the MCP server."""
        server = FastMCP(name="Scrapling", host=host, port=port)
        server.add_tool(
            self.fetch_content,
            title="fetch_content",
            description=self.fetch_content.__doc__,
            structured_output=True,
        )
        server.run(transport="stdio" if not http else "streamable-http")
