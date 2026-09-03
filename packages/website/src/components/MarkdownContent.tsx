import { memo } from 'react';
import Markdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { cn } from '@/lib/utils';

const components: Components = {
  h1: ({ children }) => <h1 className="mb-2 text-base font-semibold">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-2 text-sm font-semibold">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-1.5 text-sm font-medium">{children}</h3>,
  h4: ({ children }) => <h4 className="mb-1 text-sm font-medium">{children}</h4>,
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="my-2 flex list-disc flex-col gap-1 pl-4">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 flex list-decimal flex-col gap-1 pl-4">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="border-foreground/20 my-2 border-l-2 pl-3 italic">{children}</blockquote>
  ),
  a: ({ href, children }) => (
    <a href={href} className="underline underline-offset-2" target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  hr: () => <hr className="border-border my-3" />,
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-left text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border-border border-b px-2 py-1 font-medium">{children}</th>,
  td: ({ children }) => <td className="border-border border-b px-2 py-1">{children}</td>,
  pre: ({ children }) => (
    <pre className="bg-background my-2 overflow-x-auto rounded-md p-3 text-xs leading-relaxed">{children}</pre>
  ),
  code: ({ className, children }) =>
    className ? (
      <code className={cn('font-mono', className)}>{children}</code>
    ) : (
      <code className="bg-background rounded-sm px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
    ),
};

export const MarkdownContent = memo(function MarkdownContent({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  return (
    <div className={cn('break-words [&>*:first-child]:mt-0 [&>*:last-child]:mb-0', className)}>
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </Markdown>
    </div>
  );
});
