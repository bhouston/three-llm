import { createFileRoute, useNavigate } from '@tanstack/react-router';

import { ChatApp } from '../components/ChatApp';

type Search = {
  model?: string;
};

export const Route = createFileRoute('/')({
  ssr: false,
  validateSearch: (search: Record<string, unknown>): Search => ({
    model: typeof search.model === 'string' ? search.model : undefined,
  }),
  component: HomePage,
});

function HomePage() {
  const { model } = Route.useSearch();
  const navigate = useNavigate({ from: '/' });

  return (
    <ChatApp
      modelId={model}
      onModelChange={(id) => {
        void navigate({ search: { model: id } });
      }}
    />
  );
}
