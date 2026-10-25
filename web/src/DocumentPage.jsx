import React, { lazy, Suspense, useEffect, useRef } from 'react';
import { ArrowLeft, FileText, Moon, Sun, ArrowUpRight, BookOpen, Code2 } from 'lucide-react';
import { pretty, pairedResource, resourceKind, resourceLabels } from './model.js';
import './document-page.css';

const Markdown = lazy(() => import('./Markdown.jsx'));

export default function DocumentPage({ document: doc, path, documents, onBack, onOpen, dark, onToggleTheme, error }) {
  const heading = useRef(null);
  const title = pretty(path.split('/').at(-1).replace(/\.(md|py|ipynb)$/, ''));
  const kind = doc ? resourceKind(doc) : 'markdown';
  const pair = doc && pairedResource(documents, doc);
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' });
    heading.current?.focus({ preventScroll: true });
    const previousTitle = document.title;
    document.title = title + ' · Bloom';
    return () => { document.title = previousTitle; };
  }, [path, title]);
  const markdownComponents = {
    a: ({ href, children }) => {
      if (/^https?:\/\//i.test(href || '')) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
      let target, resolved;
      try { target = new URL(href || '', 'https://local.invalid/' + path); resolved = decodeURIComponent(target.pathname.slice(1)); }
      catch { return <span>{children}</span>; }
      if (target.origin === 'https://local.invalid' && resolved !== path && documents.some(item => item.path === resolved)) return <button className="inline-link" onClick={() => onOpen(resolved)}>{children}</button>;
      return <span title={href}>{children}</span>;
    },
    img: ({ alt }) => <span className="muted">[Image: {alt || 'reference image'}]</span>,
  };
  const markdown = content => <Suspense fallback={<p className="muted" role="status">Opening your learning material…</p>}><Markdown components={markdownComponents}>{content}</Markdown></Suspense>;
  return <main className="document-page">
    <header className="document-toolbar">
      <button className="document-back" onClick={onBack}><ArrowLeft size={19} />Back</button>
      <span className="document-toolbar-title">{kind === 'notebook' ? <BookOpen size={17} /> : kind === 'markdown' ? <FileText size={17} /> : <Code2 size={17} />}{title}</span>
      <button className="icon-button" onClick={onToggleTheme} aria-label={dark ? 'Use light theme' : 'Use dark theme'}>{dark ? <Sun size={20} /> : <Moon size={20} />}</button>
    </header>
    <div className="document-content">
      {error && <div className="error" role="alert">{error} Showing the last successfully loaded file.</div>}
      <div className="eyebrow">{resourceLabels[kind] || 'YOUR LEARNING LIBRARY'}</div>
      <h1 ref={heading} tabIndex={-1}>{title}</h1>
      <p className="document-location">{path.split('/').join(' / ')}</p>
      {doc && kind !== 'markdown' && <div className="workspace-preview-note"><span>Read-only preview · work on this file in your notebook or code editor.</span>{pair && <button className="text-button" onClick={() => onOpen(pair.path)}>{pair.kind === 'notebook' ? 'View notebook' : 'View Python file'}<ArrowUpRight size={16} /></button>}</div>}
      {doc?.error ? <div className="notice" role="status">{doc.error}</div> : doc ? <article className="markdown" aria-label={title}>
        {kind === 'notebook' ? <div className="notebook-cells">{doc.cells?.length ? doc.cells.map((cell, index) => <section key={index} className={'notebook-cell ' + cell.cell_type} aria-label={'Cell ' + (index + 1)}><small>{cell.cell_type === 'code' ? 'Code' : cell.cell_type === 'raw' ? 'Text' : 'Notes'} · {index + 1}</small>{cell.cell_type === 'markdown' ? markdown(cell.source) : <pre><code>{cell.source}</code></pre>}</section>) : <p className="muted">This notebook has no cells yet.</p>}<p className="muted">Saved outputs are kept in your local experiment records.</p></div>
          : kind === 'python' || kind === 'test' ? <pre className="source-preview"><code>{doc.content}</code></pre> : markdown(doc.content)}
      </article> : <div className="empty"><FileText size={32} /><h2>File not found</h2><p>This file is no longer available. Use Back to return.</p></div>}
    </div>
  </main>;
}
