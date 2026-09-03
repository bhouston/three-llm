import { HeartIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

export function SiteFooter() {
  return (
    <footer>
      <Separator />
      <p className="text-muted-foreground flex items-center justify-center gap-1 px-4 py-3 text-sm">
        Built with
        <span className="inline-flex items-center">
          <HeartIcon className="size-3.5 fill-current text-destructive" aria-hidden />
          <span className="sr-only">heart</span>
        </span>
        by
        <a href="https://ben3d.ca" className="text-foreground underline underline-offset-4">
          Ben Houston
        </a>
        {import.meta.env.DEV ? (
          <Button
            className="ml-2 h-7 px-2 text-xs"
            onClick={() => {
              throw new Error('Sentry Test Error');
            }}
            size="sm"
            type="button"
            variant="outline"
          >
            Break the world
          </Button>
        ) : null}
      </p>
    </footer>
  );
}
