"""
playwright_fetcher.py — Fetcher via Playwright headless=False + Xvfb.

Usado quando SOKKER_FETCH_MODE=playwright para contornar o bloqueio 403
que o Cloudflare/WAF aplica a IPs de datacenter (AWS EC2).

Execução no AWS:
    export SOKKER_FETCH_MODE=playwright
    source ~/pwenv/bin/activate
    xvfb-run -a python run.py
"""

import threading
import time
from typing import Optional
from loguru import logger

# BASE_URL do SokkerPRO — usado para filtrar respostas XHR
_SOKKER_BASE = "https://m2.sokkerpro.com"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Lock global: garante apenas 1 instância de browser rodando por vez
# (evita abrir múltiplos Chromiums em paralelo num hobby server)
_browser_lock = threading.Lock()


def fetch_json_with_playwright(url: str) -> Optional[dict | list]:
    """
    Abre o Chromium headed (headless=False) e navega até `url`.
    Captura a primeira resposta XHR com application/json que bata no endpoint.
    Retorna o JSON parseado ou None em caso de falha.

    Estratégia dupla:
    1. Escutar page.on("response") e capturar XHR JSON correspondente ao path.
    2. Fallback: tentar page.request.get(url) no contexto do browser.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    captured: dict = {"data": None}

    # Extrai o path do URL para matching (ex: "/livescores")
    path_hint = url.replace(_SOKKER_BASE, "").split("?")[0]

    def _on_response(response):
        """Callback chamado para cada resposta da página."""
        if captured["data"] is not None:
            return  # já temos o que precisamos

        try:
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return

            resp_url = response.url
            # Filtra respostas que correspondam ao endpoint esperado
            if path_hint not in resp_url and _SOKKER_BASE not in resp_url:
                return

            # Só aceita respostas bem-sucedidas
            if response.status < 200 or response.status >= 300:
                logger.warning(
                    f"[Playwright] XHR {resp_url} retornou status {response.status}"
                )
                return

            json_data = response.json()
            captured["data"] = json_data
            logger.debug(
                f"[Playwright] XHR capturado: {resp_url} | "
                f"status={response.status} | tipo={type(json_data).__name__}"
            )
        except Exception as e:
            logger.debug(f"[Playwright] erro ao processar resposta XHR: {e}")

    with _browser_lock:
        try:
            logger.info(f"[Playwright] abrindo browser para: {url}")
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=False,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
                context = browser.new_context(
                    user_agent=_USER_AGENT,
                    viewport={"width": 1280, "height": 800},
                    locale="pt-BR",
                )
                page = context.new_page()

                # Registra listener ANTES de navegar
                page.on("response", _on_response)

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                except PWTimeout:
                    logger.warning(f"[Playwright] timeout ao carregar {url}")

                # Aguarda até 8s para Cloudflare/JS assentar e XHR ser capturado
                deadline = time.time() + 8
                while time.time() < deadline:
                    if captured["data"] is not None:
                        break
                    time.sleep(0.3)

                # Fallback: se não capturou XHR, tenta request direto no contexto
                if captured["data"] is None:
                    logger.debug(
                        f"[Playwright] XHR não capturado, tentando page.request.get({url})"
                    )
                    try:
                        resp = page.request.get(
                            url,
                            headers={
                                "Accept": "application/json",
                                "User-Agent": _USER_AGENT,
                            },
                            timeout=15_000,
                        )
                        if resp.ok:
                            captured["data"] = resp.json()
                            logger.debug(
                                f"[Playwright] fallback request OK: status={resp.status}"
                            )
                        else:
                            logger.warning(
                                f"[Playwright] fallback request falhou: status={resp.status}"
                            )
                    except Exception as e:
                        logger.warning(f"[Playwright] fallback request erro: {e}")

                context.close()
                browser.close()

        except Exception as e:
            logger.error(f"[Playwright] erro crítico ao abrir browser: {e}")
            return None

    if captured["data"] is None:
        logger.warning(f"[Playwright] nenhum JSON capturado para: {url}")

    return captured["data"]
