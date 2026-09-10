import { createFileRoute, useNavigate } from '@tanstack/react-router';

import { ChatApp } from '../components/ChatApp';

type Precision = 'fp32' | 'fp16';

type Search = {
  model?: string;
  precision?: Precision;
};

export const Route = createFileRoute('/')({
  ssr: false,
  validateSearch: (search: Record<string, unknown>): Search => ({
    model: typeof search.model === 'string' ? search.model : undefined,
    precision: search.precision === 'fp32' || search.precision === 'fp16' ? search.precision : undefined,
  }),
  component: HomePage,
});

function HomePage() {
  const { model, precision } = Route.useSearch();
  const navigate = useNavigate({ from: '/' });

  return (
    <ChatApp
      modelId={model}
      onModelChange={(id) => {
        void navigate({ search: { model: id } });
      }}
      precision={precision}
      onPrecisionChange={(nextPrecision) => {
        // Precision changes the GPU weight upload format, so reload with the new
        // query parameter to reinitialize cleanly rather than hot-swapping state.
        const url = new URL(window.location.href);
        url.searchParams.set('precision', nextPrecision);
        window.location.href = url.toString();
      }}
    />
  );
}
