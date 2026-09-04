import '../app.css';
import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';
import { ThemeProvider } from 'next-themes';
import { GoogleAnalytics } from 'tanstack-router-ga4';

import { SiteFooter } from '@/components/SiteFooter';
import { SiteHeader } from '@/components/SiteHeader';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';

const GA_MEASUREMENT_ID = 'G-72GEYG4JGH';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'Three-LLM — WebGPU LLM Inference Engine' },
      {
        name: 'description',
        content:
          'The open-source Three.js-based modern LLM Inference Engine that runs in your browser via WebGPU compute.',
      },
    ],
  }),
  shellComponent: RootDocument,
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="flex h-svh flex-col">
      <SiteHeader />
      <div className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </div>
      <SiteFooter />
    </div>
  );
}

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body className="bg-background text-foreground min-h-svh antialiased">
        <GoogleAnalytics measurementId={GA_MEASUREMENT_ID} />
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <TooltipProvider>
            {children}
            <Toaster />
          </TooltipProvider>
        </ThemeProvider>
        <Scripts />
      </body>
    </html>
  );
}
