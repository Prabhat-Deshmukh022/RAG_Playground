'use client';

import React from 'react';
import { JSX } from 'react';

interface Props {
  text: string;
}

export default function FormattedLLMResponse({ text }: Props) {
  const lines = text.split('\n').map(line => line.trim());

  // Helper to parse inline **bold** text
  function parseInlineBold(text: string) {
    // Split by **bold** parts
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return <React.Fragment key={i}>{part}</React.Fragment>;
    });
  }

  // State machine for list parsing
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];
  let inList = false;

  function flushList() {
    if (listItems.length) {
      elements.push(
        <ul className="list-disc list-inside mb-4" key={`list-${elements.length}`}>
          {listItems.map((item, i) => (
            <li key={i}>{parseInlineBold(item)}</li>  // parse inline bold here
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  }

  lines.forEach((line, i) => {
    if (!line) {
      // empty line - flush list if any
      flushList();
      elements.push(<br key={`br-${i}`} />);
      return;
    }

    // Detect numbered list lines (e.g. "1. text")
    if (/^\d+\.\s/.test(line)) {
      if (!inList) inList = true;
      listItems.push(line.replace(/^\d+\.\s/, ''));
    }
    // Detect bullet list lines ("- text" or "* text")
    else if (/^[-*]\s/.test(line)) {
      if (!inList) inList = true;
      listItems.push(line.replace(/^[-*]\s/, ''));
    }
    // Detect headings ("## " or "### " etc)
    else if (/^#{1,6}\s/.test(line)) {
      flushList();
      const level = line.match(/^#{1,6}/)![0].length;
      const content = line.replace(/^#{1,6}\s/, '');
      const Tag = `h${level}` as keyof JSX.IntrinsicElements;
      elements.push(
        <Tag className="font-semibold mt-6 mb-2" key={`heading-${i}`}>
          {parseInlineBold(content)}  {/* parse inline bold here */}
        </Tag>
      );
    }
    // Detect numbered section headings like "1. Types of AI Product Managers and Their Roles"
    else if (/^\d+\.\s/.test(line)) {
      flushList();
      elements.push(
        <h2 className="font-semibold mt-6 mb-2" key={`section-${i}`}>
          {parseInlineBold(line)}  {/* parse inline bold here */}
        </h2>
      );
    } else {
      flushList();
      elements.push(
        <p className="mb-4 leading-relaxed" key={`p-${i}`}>
          {parseInlineBold(line)}  {/* parse inline bold here */}
        </p>
      );
    }
  });

  flushList();

  return <div className="prose prose-invert max-w-none">{elements}</div>;
}
