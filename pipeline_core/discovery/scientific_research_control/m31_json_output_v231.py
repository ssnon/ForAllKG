#!/usr/bin/env python3
"""M31 science-first output transport v231. JSON_OBJECT request on all LLM calls.

Does not alter scientific text, choose probes, or repair malformed outputs.
No external import side effect besides patching the native _make_call_request
once for this campaign process. Invocation receipts include the full request.
"""
from __future__ import annotations
from functools import wraps

FORMAT = {"type": "json_object"}
MARKER = "M31_JSON_OBJECT_V231"

def attach_json_mode():
    """Return an immutable JSON-output request fragment, computed without providers."""
    return {'response_format':dict(FORMAT)}

def install_native_json_mode(native):
    if getattr(native._make_call_request,'_m31_json_v231',False):
        return
    original=native._make_call_request
    @wraps(original)
    def wrapped(*args,**kwargs):
        req=original(*args,**kwargs)
        if not isinstance(req,dict) or 'messages' not in req:
            raise ValueError('NATIVE_REQUEST_SCHEMA_UNEXPECTED_V231')
        if 'response_format' in req and req['response_format']!=FORMAT:
            raise ValueError('CONFLICTING_PROVIDER_FORMAT_V231')
        new=dict(req)
        new['response_format']=dict(FORMAT)
        return new
    wrapped._m31_json_v231=True
    native._make_call_request=wrapped
