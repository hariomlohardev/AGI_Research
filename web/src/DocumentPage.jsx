import React, { lazy, Suspense, useEffect, useRef } from 'react';
import { ArrowLeft, FileText, Moon, Sun } from 'lucide-react';
import { pretty } from './model.js';
import './document-page.css';

const Markdown = lazy(() => import('./Markdown.jsx'));

export default function DocumentPage({ document: doc, path, documents, onBack, onOpen, dark, onToggleTheme, error }) {
  const heading = useRef(null);
  const title = pretty(path.split('/').at(-1).replace(/\.md$/, ''));
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' });
    heading.current?.focus({ preventScroll: true });
    const previousTitle = document.title;
    document.title = `${title} · Bloom`;
    return () => { document.title = previousTitle; };
  }, [path, title]);

  return <main className="document-page">
    <header className="document-toolbar">
      <button className="document-back" onClick={onBack}><ArrowLeft size={19} />Back</button>
      <span className="document-toolbar-title"><FileText size={17} />{title}</span>
      <button className="icon-button" onClick={onToggleTheme} aria-label={dark ? 'Use light theme' : 'Use dark theme'}>{dark ? <Sun size={20} /> : <Moon size={20} />}</button>
    </header>
    <div className="document-content">
      {error && <div className="error" role="alert">{error} Showing the last successfully loaded document.</div>}
      <div className="eyebrow">YOUR LEARNING LIBRARY</div>
      <h1 ref={heading} tabIndex={-1}>{title}</h1>
      <p className="document-location">{path.split('/').join(' / ')}</p>
      {doc ? <article className="markdown" aria-label={title}>
        <Suspense fallback={<p className="muted" role="status">Opening your learning material…</p>}>
          <Markdown components={{
            a: ({ href, children }) => {
              if (/^https?:\/\//i.test(href || '')) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
              let target, resolved;
              try {
                target = new URL(href || '', `https://local.invalid/${path}`);
                resolved = decodeURIComponent(target.pathname.slice(1));
              } catch { return <span>{children}</span>; }
              if (target.origin === 'https://local.invalid' && resolved !== path && documents.some(item => item.path === resolved)) return <button className="inline-link" onClick={() => onOpen(resolved)}>{children}</button>;
              return <span title={href}>{children}</span>;
            },
            img: ({ alt }) => <span className="muted">[Image: {alt || 'reference image'}]</span>,
          }}>{doc.content}</Markdown>
        </Suspense>
      </article> : <div className="empty"><FileText size={32} /><h2>Document not found</h2><p>This file is no longer available in your learning library. Use Back to return.</p></div>}
    </div>
  </main>;
}
