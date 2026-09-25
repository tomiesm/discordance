"""
Gene annotation fetching for Phase 3 predictability regression.

Fetches subcellular localization (UniProt) and GO Slim functional categories.
Results are cached to disk.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
import time

logger = logging.getLogger(__name__)


def fetch_uniprot_localization(gene_list: List[str],
                               cache_path: Optional[str] = None,
                               organism_id: int = 9606) -> pd.DataFrame:
    """Fetch subcellular localization from UniProt REST API.

    Args:
        gene_list: Gene symbols to query
        cache_path: Path to cache CSV. If exists, loads from cache.
        organism_id: NCBI taxonomy ID (9606 = human)

    Returns:
        DataFrame with: gene, primary_localization, all_localizations
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached UniProt data from {cache_path}")
        cached = pd.read_csv(cache_path)
        # Return only genes in gene_list
        return cached[cached['gene'].isin(gene_list)].reset_index(drop=True)

    import requests

    records = []
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    for gene in gene_list:
        params = {
            'query': f'gene_exact:{gene} AND organism_id:{organism_id}',
            'format': 'json',
            'fields': 'gene_names,cc_subcellular_location',
            'size': 1,
        }

        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get('results'):
                result = data['results'][0]
                # Extract subcellular location
                locations = []
                for comment in result.get('comments', []):
                    if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                        for loc in comment.get('subcellularLocations', []):
                            loc_val = loc.get('location', {}).get('value', '')
                            if loc_val:
                                locations.append(loc_val)

                primary = _classify_localization(locations)
                records.append({
                    'gene': gene,
                    'primary_localization': primary,
                    'all_localizations': '; '.join(locations) if locations else 'Unknown',
                })
            else:
                records.append({
                    'gene': gene,
                    'primary_localization': 'Unknown',
                    'all_localizations': 'Unknown',
                })

            # Rate limiting
            time.sleep(0.2)

        except Exception as e:
            logger.warning(f"UniProt query failed for {gene}: {e}")
            records.append({
                'gene': gene,
                'primary_localization': 'Unknown',
                'all_localizations': str(e),
            })

    df = pd.DataFrame(records)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached UniProt data to {cache_path}")

    return df


def _classify_localization(locations: List[str]) -> str:
    """Classify subcellular location into broad categories."""
    loc_str = ' '.join(locations).lower()

    if 'secreted' in loc_str or 'extracellular' in loc_str:
        return 'Secreted/Extracellular'
    elif 'membrane' in loc_str or 'cell surface' in loc_str:
        return 'Membrane'
    elif 'nucleus' in loc_str:
        return 'Nucleus'
    elif 'cytoplasm' in loc_str or 'cytosol' in loc_str:
        return 'Cytoplasm'
    elif 'mitochondri' in loc_str:
        return 'Mitochondria'
    elif 'endoplasmic reticulum' in loc_str or 'golgi' in loc_str:
        return 'ER/Golgi'
    elif locations:
        return 'Other'
    else:
        return 'Unknown'


def fetch_go_slim(gene_list: List[str],
                  cache_path: Optional[str] = None) -> pd.DataFrame:
    """Fetch GO Slim functional categories.

    Attempts to use BioMart REST API. Falls back to a simple keyword-based
    classification if the API is unavailable.

    Args:
        gene_list: Gene symbols
        cache_path: Path to cache CSV

    Returns:
        DataFrame with: gene, primary_function, all_functions
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached GO data from {cache_path}")
        cached = pd.read_csv(cache_path)
        return cached[cached['gene'].isin(gene_list)].reset_index(drop=True)

    try:
        df = _fetch_go_from_biomart(gene_list)
    except Exception as e:
        logger.warning(f"BioMart GO query failed: {e}. Using keyword classification.")
        df = _classify_genes_by_name(gene_list)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached GO data to {cache_path}")

    return df


def _fetch_go_from_biomart(gene_list: List[str]) -> pd.DataFrame:
    """Query Ensembl BioMart for GO Slim categories."""
    import requests

    # BioMart XML query
    genes_filter = ','.join(gene_list)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Query virtualSchemaName="default" formatter="TSV" header="1">
        <Dataset name="hsapiens_gene_ensembl" interface="default">
            <Filter name="hgnc_symbol" value="{genes_filter}"/>
            <Attribute name="hgnc_symbol"/>
            <Attribute name="namespace_1003"/>
        </Dataset>
    </Query>"""

    url = "http://www.ensembl.org/biomart/martservice"
    resp = requests.post(url, data={'query': xml}, timeout=30)
    resp.raise_for_status()

    lines = resp.text.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("Empty BioMart response")

    records = []
    gene_functions = {}
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) >= 2:
            gene, go_term = parts[0], parts[1]
            if gene not in gene_functions:
                gene_functions[gene] = []
            if go_term:
                gene_functions[gene].append(go_term)

    for gene in gene_list:
        functions = gene_functions.get(gene, [])
        primary = _classify_go_terms(functions) if functions else 'Unknown'
        records.append({
            'gene': gene,
            'primary_function': primary,
            'all_functions': '; '.join(functions[:5]) if functions else 'Unknown',
        })

    return pd.DataFrame(records)


def _classify_go_terms(terms: List[str]) -> str:
    """Classify GO terms into broad functional categories."""
    terms_str = ' '.join(terms).lower()

    if any(x in terms_str for x in ['kinase', 'phosphatase', 'enzyme', 'catalytic']):
        return 'Enzymatic'
    elif any(x in terms_str for x in ['receptor', 'ligand binding']):
        return 'Receptor'
    elif any(x in terms_str for x in ['transcription factor', 'dna binding', 'transcription']):
        return 'Transcription factor'
    elif any(x in terms_str for x in ['transport', 'channel', 'carrier']):
        return 'Transporter'
    elif any(x in terms_str for x in ['structural', 'cytoskeleton', 'extracellular matrix']):
        return 'Structural'
    elif any(x in terms_str for x in ['signal', 'signaling']):
        return 'Signaling'
    else:
        return 'Other'


def _classify_genes_by_name(gene_list: List[str]) -> pd.DataFrame:
    """Simple keyword-based gene classification as fallback."""
    records = []
    for gene in gene_list:
        g = gene.upper()
        if g.startswith('COL') or g.startswith('KRT') or g.startswith('ACT'):
            primary = 'Structural'
        elif g.startswith('CD') or g.startswith('HLA'):
            primary = 'Receptor/Immune'
        elif g.startswith('IL') or g.startswith('CXCL') or g.startswith('CCL'):
            primary = 'Signaling/Cytokine'
        elif g.startswith('MMP') or g.startswith('ADAM'):
            primary = 'Enzymatic'
        elif g.startswith('SLC') or g.startswith('ABC'):
            primary = 'Transporter'
        else:
            primary = 'Other'

        records.append({
            'gene': gene,
            'primary_function': primary,
            'all_functions': primary,
        })

    return pd.DataFrame(records)
