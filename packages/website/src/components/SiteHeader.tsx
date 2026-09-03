import { BookOpenIcon, GithubIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/** Official Three.js mark from `files/icon.svg`, using currentColor for theming. */
function ThreeJsIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 226.77 226.77" fill="none" aria-hidden {...props}>
      <g
        transform="translate(8.964 4.2527)"
        fillRule="evenodd"
        stroke="currentColor"
        strokeLinecap="butt"
        strokeLinejoin="round"
        strokeWidth="4"
      >
        <path d="m63.02 200.61-43.213-174.94 173.23 49.874z" />
        <path d="m106.39 50.612 21.591 87.496-86.567-24.945z" />
        <path d="m84.91 125.03-10.724-43.465 43.008 12.346z" />
        <path d="m63.458 38.153 10.724 43.465-43.008-12.346z" />
        <path d="m149.47 62.93 10.724 43.465-43.008-12.346z" />
        <path d="m84.915 125.06 10.724 43.465-43.008-12.346z" />
      </g>
    </svg>
  );
}

function NpmIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden {...props}>
      <path d="M0 0v24h24V0H0zm19.2 19.2h-4.8V8.4H9.6v10.8H4.8V4.8h14.4v14.4z" />
    </svg>
  );
}

export function SiteHeader() {
  return (
    <header className="border-b bg-background">
      <div className="mx-auto flex w-full max-w-3xl items-start justify-between gap-4 px-4 py-3">
        <div className="flex min-w-0 items-start gap-3">
          <ThreeJsIcon className="mt-0.5 size-8 shrink-0" />
          <div className="flex min-w-0 flex-col gap-0.5">
            <h1 className="font-heading text-base font-medium tracking-tight">Three-LLM Demo Chat App</h1>
            <p className="text-muted-foreground text-sm text-pretty">
              Build using ' <a href="https://github.com/bhouston/three-llm" className="text-foreground underline underline-offset-4">three-llm</a>', the open-source Three.js-based modern LLM Engine that runs in your browser via WebGPU compute.{' '}
              <a
                href="https://ben3d.ca/blog/running-llms-in-the-browser-with-threejs"
                className="text-foreground underline underline-offset-4"
              >
                Read the technical write-up
              </a>
              .
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  nativeButton={false}
                  render={
                    <a
                      href="https://ben3d.ca/blog/running-llms-in-the-browser-with-threejs"
                      aria-label="Technical write-up"
                    />
                  }
                  aria-label="Technical write-up"
                />
              }
            >
              <BookOpenIcon />
            </TooltipTrigger>
            <TooltipContent>Technical write-up</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  nativeButton={false}
                  render={<a href="https://github.com/bhouston/three-llm" aria-label="GitHub" />}
                  aria-label="GitHub"
                />
              }
            >
              <GithubIcon />
            </TooltipTrigger>
            <TooltipContent>GitHub</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  nativeButton={false}
                  render={<a href="https://www.npmjs.com/package/three-llm" aria-label="npm" />}
                  aria-label="npm"
                />
              }
            >
              <NpmIcon />
            </TooltipTrigger>
            <TooltipContent>npm</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </header>
  );
}
