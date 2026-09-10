import { CpuIcon, GithubIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

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
          <CpuIcon className="mt-0.5 size-8 shrink-0" aria-hidden />
          <div className="flex min-w-0 flex-col gap-0.5">
            <h1 className="font-heading text-base font-medium tracking-tight">vgpu-llm Demo Chat App</h1>
            <p className="text-muted-foreground text-sm text-pretty">
              Built using{' '}
              <a href="https://github.com/bhouston/vgpu-llm" className="text-primary underline underline-offset-4">
                vgpu-llm
              </a>
              , the open-source{' '}
              <a href="https://github.com/vercel-labs/vgpu" className="text-primary underline underline-offset-4">
                vgpu
              </a>
              -based modern LLM inference engine that runs in your browser via WebGPU compute.
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
                  render={<a href="https://github.com/bhouston/vgpu-llm" aria-label="GitHub" />}
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
                  render={<a href="https://www.npmjs.com/package/vgpu-llm" aria-label="npm" />}
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
