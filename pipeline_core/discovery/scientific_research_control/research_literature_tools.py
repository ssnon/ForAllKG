"""Existing FAKG OpenAlex discovery + bounded, source-depth-aware scientific reading.

No novel bibliography backend. One on-demand network operation per research step.
All fetched evidence is materialized by the caller into immutable step receipts.
"""
from __future__ import annotations
import hashlib
import io
import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

MAX_ABSTRACT = 4500
MAX_PDF_BYTES = 12 * 1024 * 1024
MAX_FULLTEXT = 16000
MAX_SEARCH_LIMIT = 7


def _sanitized_error(exc: Exception) -> str:
    # Never persist provider error strings/URLs, which may contain API keys.
    return type(exc).__name__


def _clean_record(record: Any) -> dict:
    row = record.to_dict() if hasattr(record, 'to_dict') else dict(record)
    refs = row.get('provider_references') or []
    urls = row.get('metadata') or {}
    oa = urls.get('best_oa_location') or {}
    primary = urls.get('primary_location') or {}
    return {
        'paper_id':str(row['paper_id']),
        'title':str(row['title'])[:400],
        'year':row.get('year'),
        'publication_date':urls.get('publication_date'),
        'is_retracted':urls.get('is_retracted'),
        'doi':row.get('doi'),
        'venue':row.get('venue'),
        'abstract':str(row.get('abstract') or '')[:MAX_ABSTRACT],
        'source_depth':'abstract' if row.get('abstract') else 'metadata',
        'provider_references':refs,
        'openalex_id':urls.get('openalex_id'),
        'oa_pdf_url':oa.get('pdf_url') if oa.get('is_oa') is True else None,
        'landing_page_url':oa.get('landing_page_url') or primary.get('landing_page_url'),
        'discovery_queries':list(row.get('discovery_queries') or []),
    }


def search(payload: dict, *, provider=None) -> dict:
    queries = payload.get('queries')
    if not isinstance(queries, list) or not 1 <= len(queries) <= 2:
        raise ValueError('SEARCH_REQUIRES_ONE_OR_TWO_QUERIES')
    if any(not isinstance(q,str) or not 5 <= len(q.strip()) <= 240 for q in queries):
        raise ValueError('INVALID_SEARCH_QUERY')
    limit = payload.get('limit', 5)
    if type(limit) is not int or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ValueError('INVALID_SEARCH_LIMIT')
    if provider is None:
        # Import the ORIGINAL FAKG provider, not a replacement search implementation.
        from pipeline_core.literature.discovery.providers.openalex import OpenAlexProvider
        provider = OpenAlexProvider(api_key=os.getenv('OPENALEX_API_KEY') or None,
                                    mailto=os.getenv('OPENALEX_MAILTO') or None,
                                    timeout=18, page_size=limit, max_retries=1,
                                    backoff_base_seconds=0.5)
    from pipeline_core.literature.discovery.providers.base import LiteratureSearchRequest
    results=[];seen=set();failures=[]
    for query in queries:
        try:
            req=LiteratureSearchRequest(query=query.strip(), mechanism_bucket='scientific_research',limit=limit)
            docs=provider.search(req)
            for doc in docs:
                row=_clean_record(doc)
                identity=(row['doi'] or row['paper_id']).lower()
                if identity not in seen:
                    results.append(row);seen.add(identity)
        except Exception as exc:
            failures.append({'query':query,'error_type':_sanitized_error(exc)})
    return {'status':'SEARCH_OK' if results else ('SEARCH_FAILED' if failures else 'NO_RESULTS'),
            'provider':'FAKG_OpenAlexProvider','queries':queries,'papers':results[:2*limit],
            'failures':failures,'source_depth':'metadata_or_abstract',
            'novelty_certified':False}


def _public_host(url: str) -> bool:
    p=urlparse(url)
    if p.scheme!='https' or not p.hostname or p.username or p.password or p.port not in (None,443):
        return False
    host=p.hostname.lower().rstrip('.')
    if host=='localhost' or host.endswith('.local'):
        return False
    try:
        addresses=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not addresses:return False
    for a in addresses:
        ip=ipaddress.ip_address(a[4][0])
        if not ip.is_global:return False
    return True


class _SafeRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _public_host(newurl):
            raise ValueError('NON_PUBLIC_PDF_REDIRECT')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def _extract_public_pdf(url: str, *, pdf_fetcher=None) -> dict:
    if not _public_host(url) and pdf_fetcher is None:
        raise ValueError('NON_PUBLIC_PDF_URL')
    if pdf_fetcher is None:
        opener=build_opener(_SafeRedirects())
        request=Request(url,headers={'User-Agent':'FAKG-ScientificResearch-Literature/1.0',
                                     'Accept':'application/pdf'})
        with opener.open(request,timeout=18) as response:
            buf=response.read(MAX_PDF_BYTES+1)
    else:
        buf=pdf_fetcher(url)
    if len(buf)>MAX_PDF_BYTES:raise ValueError('PDF_TOO_LARGE')
    if not buf.startswith(b'%PDF-'):raise ValueError('NOT_A_PDF')
    try:
        from pypdf import PdfReader
    except ImportError:
        return {'status':'PDF_READER_UNAVAILABLE','sha256':hashlib.sha256(buf).hexdigest()}
    reader=PdfReader(io.BytesIO(buf),strict=False)
    segments=[]
    for i,page in enumerate(reader.pages[:35],1):
        s=(page.extract_text() or '').strip()
        if s:segments.append(f'[PDF page {i}]\n{s}')
        if sum(map(len,segments))>=MAX_FULLTEXT:break
    joined='\n\n'.join(segments)
    text=joined[:MAX_FULLTEXT]
    truncated=len(joined)>MAX_FULLTEXT or i < len(reader.pages) if segments else len(reader.pages)>35
    return {'status':'FULLTEXT_EXTRACTED' if text else 'PDF_TEXT_NOT_EXTRACTABLE',
            'sha256':hashlib.sha256(buf).hexdigest(), 'page_count':len(reader.pages),
            'excerpt_truncated':bool(truncated), 'text':text,
            'source_depth':'fulltext_excerpt' if text else 'abstract'}


def read(payload: dict, *, known_papers: list[dict], provider=None, pdf_fetcher=None) -> dict:
    paper_id=payload.get('paper_id')
    depth=payload.get('depth','abstract')
    if not isinstance(paper_id,str) or depth not in ('abstract','fulltext'):
        raise ValueError('INVALID_READ_REQUEST')
    matches=[p for p in known_papers if p.get('paper_id')==paper_id]
    if len(matches)!=1:raise ValueError('READ_REQUIRES_PREVIOUSLY_DISCOVERED_PAPER_ID')
    source=matches[0]
    base={'status':'ABSTRACT_READ' if source.get('abstract') else 'METADATA_ONLY',
          'paper_id':paper_id,'title':source.get('title'),'doi':source.get('doi'),
          'source_depth':'abstract' if source.get('abstract') else 'metadata',
          'abstract':source.get('abstract') or '',
          'oa_pdf_url':source.get('oa_pdf_url'),
          'provider':'FAKG_OpenAlexProvider','novelty_certified':False}
    if depth=='abstract':return base
    if provider is None:
        from pipeline_core.literature.discovery.providers.openalex import OpenAlexProvider
        provider=OpenAlexProvider(api_key=os.getenv('OPENALEX_API_KEY') or None,
                                  mailto=os.getenv('OPENALEX_MAILTO') or None,
                                  timeout=18,max_retries=1)
    try:
        identifier=source.get('openalex_id') or source.get('doi')
        if not identifier:raise ValueError('NO_OPENALEX_OR_DOI_IDENTIFIER')
        work=provider.get_work(identifier)
        # Reuse FAKG's existing OA resolution (no paywalled scraping).
        from pipeline_core.literature.acquisition.openalex_access import _locations_from_openalex_work
        locations=_locations_from_openalex_work(work_id=paper_id,raw_work=work)
        pdfs=[x.url_for_pdf for x in locations if x.automatic_download_eligible and x.url_for_pdf]
        if not pdfs:return base|{'status':'NO_VERIFIED_OA_PDF_LOCATION'}
        detail=_extract_public_pdf(pdfs[0],pdf_fetcher=pdf_fetcher)
        if detail['status']!='FULLTEXT_EXTRACTED':
            return base|{'status':detail['status'],'pdf_sha256':detail.get('sha256')}
        return base|{'status':'FULLTEXT_READ','source_depth':'fulltext_excerpt',
                     'fulltext_excerpt':detail['text'],'page_count':detail['page_count'],
                     'pdf_sha256':detail['sha256'],'oa_pdf_url':pdfs[0],
                     'excerpt_truncated':detail['excerpt_truncated']}
    except Exception as exc:
        return base|{'status':'FULLTEXT_UNAVAILABLE','read_error_type':_sanitized_error(exc)}
