# -*- coding: utf-8 -*-
"""Page-lazy registration for heavy column-list callbacks.

With wide datasets (1000+ columns) every column picker carries ~100 KB of
options and the Dashboard alone has four sets of eleven pickers.  Filling
all of them on every file load made the browser re-render megabytes of lists
for pages the user was not even looking at.

``lazy_callback`` registers the same callback but:

* runs it only while its page is open (``url.pathname``);
* remembers a short signature of its inputs in a ``dcc.Store`` so returning to
  the page without data changes does not resend the same lists.

The decorated function itself is returned unchanged, so it can still be
called directly (tests, other helpers).
"""

from __future__ import annotations

import hashlib
import json

from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate

from dash_app import app


LAZY_KEYS: list[str] = []


def _sig_store_id(key: str) -> str:
    return f"lazy-sig-{key}"


def lazy_sig_stores():
    """dcc.Store components that must be present in the layout."""
    return [dcc.Store(id=_sig_store_id(key)) for key in LAZY_KEYS]


def page_matches(pathname, page: str) -> bool:
    if page == "/":
        return pathname in (None, "", "/")
    return pathname == page


def _signature(values) -> str:
    raw = json.dumps(values, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


def _flatten(dependencies):
    for item in dependencies:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def lazy_callback(page: str, key: str, *dependencies, **kwargs):
    """Like ``app.callback`` but only fills outputs while *page* is open."""
    if key in LAZY_KEYS:
        raise ValueError(f"Duplicate lazy callback key: {key}")
    LAZY_KEYS.append(key)
    sig_id = _sig_store_id(key)

    flat = list(_flatten(dependencies))
    outputs = [item for item in flat if isinstance(item, Output)]
    inputs = [item for item in flat if isinstance(item, Input)]
    states = [item for item in flat if isinstance(item, State)]
    single_output = len(outputs) == 1

    def decorator(func):
        @app.callback(
            *outputs,
            Output(sig_id, "data"),
            *inputs,
            Input("url", "pathname"),
            *states,
            State(sig_id, "data"),
            **kwargs,
        )
        def _lazy_wrapper(*args):
            input_values = args[: len(inputs)]
            pathname = args[len(inputs)]
            state_values = args[len(inputs) + 1 : -1]
            stored_signature = args[-1]

            if not page_matches(pathname, page):
                raise PreventUpdate
            signature = _signature([*input_values, *state_values])
            if signature == stored_signature:
                raise PreventUpdate

            result = func(*input_values, *state_values)
            if single_output:
                return result, signature
            return (*result, signature)

        return func

    return decorator
