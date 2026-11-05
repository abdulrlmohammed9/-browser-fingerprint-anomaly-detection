"""Read FP-Stalker MySQL data as text, never execute SQL."""
import re
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC = ['screen_width', 'screen_height', 'timezone_offset', 'plugin_count']
CATEGORICAL = ['platform', 'cookies', 'local_storage', 'session_storage', 'adblock', 'browser_family']
FEATURES = NUMERIC + CATEGORICAL


def tuples(text):
    """Parse scalar INSERT tuples; reject expressions and truncated records."""
    row, token, quoted, escape, active, was_quoted = [], [], False, False, False, False
    mapping = {'0': '\0', 'n': '\n', 'r': '\r', 't': '\t', 'b': '\b', 'Z': '\x1a'}
    it = iter(text)
    def value():
        s = ''.join(token)
        if was_quoted:
            return s
        s = s.strip()
        if s.upper() == 'NULL': return None
        if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s):
            raise ValueError('Unsupported SQL value (no SQL was executed)')
        return s
    for c in it:
        if quoted:
            if escape:
                token.append(mapping.get(c, c)); escape = False
            elif c == '\\': escape = True
            elif c == "'": quoted = False
            else: token.append(c)
        elif c == "'":
            # Adjacent SQL quotes encode one quote in an already quoted token.
            if was_quoted: token.append("'")
            elif ''.join(token).strip(): raise ValueError('Unsupported SQL expression')
            else: token.clear()
            quoted = True; was_quoted = True
        elif c == '(':
            if active: raise ValueError('Nested SQL expression is unsupported')
            active = True; row = []; token = []; was_quoted = False
        elif active and c in ',)':
            row.append(value()); token = []; was_quoted = False
            if c == ')':
                active = False
                yield row
        elif active:
            if was_quoted and not c.isspace(): raise ValueError('Unexpected text after SQL string')
            if not was_quoted: token.append(c)
        elif c == ';': return
        elif not c.isspace() and c != ',': raise ValueError('Unexpected INSERT syntax')
    if active or quoted or escape: raise ValueError('Truncated SQL INSERT')


def read_sql(path):
    """Handle mysqldump INSERT statements with explicit or schema column order."""
    columns, records = [], []
    required = {'creationDate', 'id', 'resolutionJS', 'pluginsJS', 'timezoneJS', 'platformJS',
                'cookiesJS', 'localJS', 'sessionJS', 'adBlock', 'userAgentHttp'}
    with Path(path).open(encoding='utf-8', errors='strict') as f:
        in_schema = False
        for line in f:
            if re.match(r'CREATE TABLE\s+`extensionDataScheme`', line, re.I):
                in_schema = True; columns = []; continue
            if in_schema:
                m = re.match(r'\s*`([^`]+)`\s+', line)
                if m: columns.append(m.group(1))
                if line.lstrip().startswith(')'): in_schema = False
                continue
            m = re.match(r'INSERT INTO\s+`extensionDataScheme`\s*(\([^;]*?\))?\s*VALUES\s*', line, re.I)
            if not m: continue
            names = re.findall(r'`([^`]+)`', m[1]) if m[1] else columns
            if not required.issubset(names): raise ValueError('Required FP-Stalker columns are missing')
            # Stream subsequent lines of the same INSERT; do not load the entire dump.
            def chars():
                rest = line[m.end():]
                yield from rest
                while not rest.rstrip().endswith(';'):
                    rest = next(f, '')
                    if not rest: break
                    yield from rest
            for values in tuples(chars()):
                if len(values) != len(names): raise ValueError('INSERT column/value count mismatch')
                # Retain only explicitly needed fields; discard addresses, hashes and images.
                records.append({k: v for k, v in zip(names, values) if k in required})
    if not records: raise ValueError('No FP-Stalker INSERT records found')
    return pd.DataFrame(records)


def prepare(raw):
    out = pd.DataFrame(index=raw.index)
    resolution = raw.resolutionJS.fillna('').str.extract(r'^\s*(\d+)\s*x\s*(\d+)(?:\s*x\s*\d+)?\s*$', flags=re.I)
    for i, name in enumerate(NUMERIC[:2]):
        out[name] = pd.to_numeric(resolution[i], errors='coerce').where(lambda x: x.between(1, 20000))
    out['timezone_offset'] = pd.to_numeric(raw.timezoneJS, errors='coerce').where(lambda x: x.between(-1440, 1440))
    p = raw.pluginsJS.fillna('')
    counts = p.str.count(r'Plugin\s+\d+\s*:').astype(float)
    # Unknown formats remain missing rather than silently inventing zero plugins.
    out['plugin_count'] = counts.where((counts > 0) | p.str.strip().eq(''))
    for source, target in [('platformJS', 'platform'), ('cookiesJS', 'cookies'), ('localJS', 'local_storage'), ('sessionJS', 'session_storage'), ('adBlock', 'adblock')]:
        out[target] = raw[source].fillna('Missing').str.strip().str.slice(0, 80).replace('', 'Missing')
    ua = raw.userAgentHttp.fillna('')
    out['browser_family'] = np.select([ua.str.contains(x, case=False, regex=True) for x in [r'Edg|Edge', r'OPR|Opera', r'Firefox', r'Chrome|Chromium', r'Safari']], ['Edge', 'Opera', 'Firefox', 'Chrome', 'Safari'], default='Other')
    out['_time'] = pd.to_datetime(raw.creationDate, errors='coerce')
    out['_browser'] = raw.id.fillna('').astype(str)
    return out


def chronological_split(frame):
    df = frame.dropna(subset=['_time']).sort_values('_time', kind='stable').copy()
    times = df['_time'].drop_duplicates().to_numpy()
    if len(times) < 5: raise ValueError('Need at least five distinct timestamps')
    a, b = times[int(len(times)*.6)], times[int(len(times)*.8)]
    fit, cal, test = df[df._time < a], df[(df._time >= a) & (df._time < b)], df[df._time >= b]
    if min(map(len, (fit, cal, test))) < 10: raise ValueError('Need at least 10 records in every chronological split')
    return fit, cal, test
