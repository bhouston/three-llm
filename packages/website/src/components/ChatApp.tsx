import { catalogLabel, DEFAULT_MODEL_ID, MODEL_CATALOG, resolveModelURL } from 'three-llm/catalog';
import type { ChatMessage, GenerateOptions, GenerationResult, ModelCatalogEntry } from 'three-llm';
import { ArrowUpIcon, MessageSquareIcon, SquareIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';

import { MarkdownContent } from '@/components/MarkdownContent';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty';
import { Field, FieldDescription, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea } from '@/components/ui/input-group';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Spinner } from '@/components/ui/spinner';
import { Switch } from '@/components/ui/switch';

type TslRunner = {
  maxTokens: number;
  generate: (renderer: unknown, prompt: string, options: GenerateOptions) => Promise<GenerationResult>;
  resetCache: () => void;
  weights: {
    architecture: string;
    tokenizer: { decode: (ids: number[]) => string; encode: (text: string) => number[] };
    formatChat?: (messages: ChatMessage[], options?: { enableThinking?: boolean }) => string;
  };
};

type ChatTurn = { role: 'user' | 'assistant'; text: string };

const DEFAULT_MAX_NEW_TOKENS = 1024;
const DRAFT_FLUSH_MS = 1000;

function ChatBubble({ turn, streaming = false }: { turn: ChatTurn; streaming?: boolean }) {
  const isUser = turn.role === 'user';
  return (
    <div
      className={
        isUser
          ? 'bg-primary text-primary-foreground ml-auto max-w-[85%] rounded-lg px-3 py-2 whitespace-pre-wrap'
          : 'bg-muted max-w-[85%] rounded-lg px-3 py-2'
      }
    >
      {isUser ? turn.text : <MarkdownContent text={turn.text} />}
      {streaming ? (
        <span className="bg-foreground/70 mt-2 inline-block size-1.5 animate-pulse rounded-full" aria-hidden />
      ) : null}
    </div>
  );
}

function selectedModel(modelId: string | undefined): ModelCatalogEntry {
  return (
    MODEL_CATALOG.find((entry) => entry.id === modelId) ??
    MODEL_CATALOG.find((entry) => entry.id === DEFAULT_MODEL_ID) ??
    MODEL_CATALOG[0]!
  );
}

function defaultMaxNewTokens(contextLimit: number): number {
  return Math.max(1, Math.min(DEFAULT_MAX_NEW_TOKENS, contextLimit - 1));
}

function conversationPrompt(runner: TslRunner, turns: ChatTurn[], enableThinking: boolean) {
  if (runner.weights.formatChat) {
    return runner.weights.formatChat(turns, { enableThinking });
  }

  let prompt = '';
  for (const turn of turns) {
    const label = turn.role === 'user' ? 'User' : 'Assistant';
    prompt += `${label}: ${turn.text}\n\n`;
  }
  return `${prompt}Assistant:`;
}

export function ChatApp({ modelId, onModelChange }: { modelId?: string; onModelChange: (id: string) => void }) {
  const model = useMemo(() => selectedModel(modelId), [modelId]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const runnerRef = useRef<TslRunner | undefined>(undefined);
  const rendererRef = useRef<{ dispose: () => void } | undefined>(undefined);
  const conversationTokensRef = useRef<number[] | undefined>(undefined);
  const abortRef = useRef<AbortController | undefined>(undefined);
  const assistantDraftLatestRef = useRef('');
  const draftFlushTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const lastDraftFlushRef = useRef(0);

  const [status, setStatus] = useState('Checking WebGPU…');
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState(model.prompt);
  const [assistantDraft, setAssistantDraft] = useState('');
  const [generating, setGenerating] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULT_MAX_NEW_TOKENS);
  const [contextLimit, setContextLimit] = useState(DEFAULT_MAX_NEW_TOKENS);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(10);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [noRepeatNgramSize, setNoRepeatNgramSize] = useState(3);
  const [enableThinking, setEnableThinking] = useState(false);
  const [architecture, setArchitecture] = useState<string | null>(null);
  const [tokenRate, setTokenRate] = useState<string>('');

  const isQwen = (architecture ?? model.id).includes('qwen');

  const clearDraftFlushTimer = useCallback(() => {
    if (draftFlushTimerRef.current !== undefined) {
      clearTimeout(draftFlushTimerRef.current);
      draftFlushTimerRef.current = undefined;
    }
  }, []);

  const flushAssistantDraft = useCallback(() => {
    clearDraftFlushTimer();
    lastDraftFlushRef.current = performance.now();
    setAssistantDraft(assistantDraftLatestRef.current);
  }, [clearDraftFlushTimer]);

  const resetAssistantDraft = useCallback(() => {
    assistantDraftLatestRef.current = '';
    lastDraftFlushRef.current = 0;
    clearDraftFlushTimer();
    setAssistantDraft('');
  }, [clearDraftFlushTimer]);

  const queueAssistantDraft = useCallback(
    (text: string) => {
      assistantDraftLatestRef.current = text;
      const elapsed = performance.now() - lastDraftFlushRef.current;
      if (elapsed >= DRAFT_FLUSH_MS) {
        flushAssistantDraft();
        return;
      }
      if (draftFlushTimerRef.current === undefined) {
        draftFlushTimerRef.current = setTimeout(flushAssistantDraft, DRAFT_FLUSH_MS - elapsed);
      }
    },
    [flushAssistantDraft],
  );

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [history, assistantDraft, scrollToBottom]);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      setReady(false);
      setError(null);
      setHistory([]);
      setDraft(model.prompt);
      assistantDraftLatestRef.current = '';
      lastDraftFlushRef.current = 0;
      if (draftFlushTimerRef.current !== undefined) {
        clearTimeout(draftFlushTimerRef.current);
        draftFlushTimerRef.current = undefined;
      }
      setAssistantDraft('');
      conversationTokensRef.current = undefined;
      setArchitecture(null);

      if (typeof navigator === 'undefined' || navigator.gpu === undefined) {
        setError('This browser does not support WebGPU. Try a recent Chrome, Edge, or Safari.');
        setStatus('WebGPU unavailable');
        return;
      }

      const canvas = canvasRef.current;
      if (!canvas) return;

      try {
        setStatus(`Resolving ${model.name}…`);
        const [{ WebGPURenderer }, { createTSLRunner }] = await Promise.all([
          import('three/webgpu'),
          import('three-llm'),
        ]);

        const renderer = new WebGPURenderer({ canvas, antialias: false });
        await renderer.init();
        rendererRef.current = renderer;

        const modelURL = await resolveModelURL(model);
        const fromLocal = model.localUrl !== undefined && modelURL === model.localUrl;
        setStatus(`Loading ${model.name} from ${fromLocal ? 'hosted models' : modelURL}…`);

        const runner = (await createTSLRunner(modelURL, {
          prefillChunkSize: 4,
          onProgress: (message) => {
            if (!cancelled) setStatus(message);
          },
        })) as TslRunner;

        if (cancelled) {
          runnerRef.current = undefined;
          renderer.dispose();
          return;
        }

        runnerRef.current = runner;
        setArchitecture(runner.weights.architecture);
        setContextLimit(runner.maxTokens);
        setMaxNewTokens(defaultMaxNewTokens(runner.maxTokens));
        setReady(true);
        setStatus(`Ready. Context ${runner.maxTokens} tokens.`);
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : String(loadError);
        if (!cancelled) {
          setError(message);
          setStatus(`Load failed: ${message}`);
          toast.error('Failed to load model', { description: message });
        }
      }
    }

    void init();

    return () => {
      cancelled = true;
      abortRef.current?.abort();
      rendererRef.current?.dispose();
      rendererRef.current = undefined;
      runnerRef.current = undefined;
    };
  }, [model]);

  useEffect(() => () => clearDraftFlushTimer(), [clearDraftFlushTimer]);

  function clearChat() {
    abortRef.current?.abort();
    setHistory([]);
    resetAssistantDraft();
    conversationTokensRef.current = undefined;
    runnerRef.current?.resetCache();
    if (runnerRef.current) setStatus(`Ready. Context ${runnerRef.current.maxTokens} tokens.`);
  }

  async function sendMessage() {
    const runner = runnerRef.current;
    const renderer = rendererRef.current;
    const userText = draft.trim();
    if (!userText || !runner || !renderer || generating) return;

    const abortController = new AbortController();
    abortRef.current = abortController;
    setGenerating(true);
    setDraft('');
    resetAssistantDraft();
    setTokenRate('');

    const nextHistory: ChatTurn[] = [...history, { role: 'user', text: userText }];
    setHistory(nextHistory);
    setStatus('Generating…');

    const generatedIds: number[] = [];
    const generationStart = performance.now();

    try {
      const tokenBudget = Math.min(Math.max(1, maxNewTokens), Math.max(1, runner.maxTokens - 1));
      const generateOptions: GenerateOptions = {
        maxNewTokens: tokenBudget,
        temperature,
        topK,
        repetitionPenalty,
        noRepeatNgramSize: Math.max(0, noRepeatNgramSize),
        signal: abortController.signal,
        onPrefill: ({ cachedPromptTokens, promptTokens }) => {
          const fresh = promptTokens - cachedPromptTokens;
          setStatus(
            cachedPromptTokens > 0
              ? `Prefilling ${fresh} new prompt token(s); ${cachedPromptTokens} cached…`
              : `Prefilling ${fresh} prompt token(s)…`,
          );
        },
        onPrefillProgress: ({ cachedPromptTokens, completedPromptTokens = 0, promptTokens }) => {
          const fresh = promptTokens - cachedPromptTokens;
          const completed = Math.max(0, completedPromptTokens - cachedPromptTokens);
          setStatus(
            cachedPromptTokens > 0
              ? `Prefilling ${Math.min(completed, fresh)}/${fresh} new prompt token(s); ${cachedPromptTokens} cached…`
              : `Prefilling ${completedPromptTokens}/${promptTokens} prompt token(s)…`,
          );
        },
        onToken: (_text, nextToken) => {
          generatedIds.push(nextToken);
          queueAssistantDraft(runner.weights.tokenizer.decode(generatedIds));
          const elapsed = (performance.now() - generationStart) / 1000;
          if (generatedIds.length > 0 && elapsed > 0) {
            setTokenRate(`${(generatedIds.length / elapsed).toFixed(1)} tok/s`);
            setStatus(`Generating… ${(generatedIds.length / elapsed).toFixed(1)} tok/s`);
          }
        },
      };

      if (conversationTokensRef.current && conversationTokensRef.current.length > 0 && !runner.weights.formatChat) {
        generateOptions.inputTokens = conversationTokensRef.current.concat(
          runner.weights.tokenizer.encode(`\n\nUser: ${userText}\n\nAssistant:`),
        );
      }

      const result = await runner.generate(
        renderer,
        conversationPrompt(runner, nextHistory, enableThinking),
        generateOptions,
      );
      const reply = result.generatedText || runner.weights.tokenizer.decode(generatedIds);
      conversationTokensRef.current = result.tokens;
      setHistory([...nextHistory, { role: 'assistant', text: reply }]);
      resetAssistantDraft();

      const cacheNote = result.cachedPromptTokens
        ? ` Cached ${result.cachedPromptTokens}/${result.promptTokens} prompt tokens.`
        : '';
      const elapsed = (performance.now() - generationStart) / 1000;
      const rate =
        result.generatedTokens.length > 0 && elapsed > 0
          ? ` ${(result.generatedTokens.length / elapsed).toFixed(1)} tok/s`
          : '';
      setTokenRate(rate.trim());
      setStatus(
        result.aborted
          ? `Stopped after ${result.generatedTokens.length} token(s).${rate}${cacheNote}`
          : `Generated ${result.generatedTokens.length} token(s).${rate}${cacheNote}`,
      );
    } catch (generateError) {
      const message = generateError instanceof Error ? generateError.message : String(generateError);
      setStatus(message);
      toast.error('Generation failed', { description: message });
      if (assistantDraftLatestRef.current !== '') {
        setHistory([...nextHistory, { role: 'assistant', text: assistantDraftLatestRef.current }]);
      }
      resetAssistantDraft();
    } finally {
      abortRef.current = undefined;
      setGenerating(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <canvas ref={canvasRef} className="hidden" width={1} height={1} aria-hidden />
      <header className="border-b bg-background">
        <div className="mx-auto flex w-full max-w-5xl items-center justify-between gap-4 px-4 py-4">
          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium">three-llm</p>
            <p className="text-muted-foreground text-sm">WebGPU LLM chat in the browser</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" nativeButton={false} render={<a href="https://github.com/bhouston/three-llm" />}>
              GitHub
            </Button>
            <Button
              variant="outline"
              nativeButton={false}
              render={<a href="https://www.npmjs.com/package/three-llm" />}
            >
              npm
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4 px-4 py-6">
        <Card>
          <CardHeader>
            <CardTitle>Model</CardTitle>
            <CardDescription>{model.note}</CardDescription>
          </CardHeader>
          <CardContent>
            <FieldGroup className="gap-4">
              <div className="flex flex-col gap-4 md:flex-row md:items-end">
                <Field className="flex-1">
                  <FieldLabel>Checkpoint</FieldLabel>
                  <Select
                    value={model.id}
                    onValueChange={(value) => {
                      if (typeof value === 'string') onModelChange(value);
                    }}
                    disabled={generating}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue>{catalogLabel(model)}</SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {MODEL_CATALOG.map((entry) => (
                          <SelectItem key={entry.id} value={entry.id}>
                            {catalogLabel(entry)}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <FieldDescription>Weights stream from the hosted model bucket on first load.</FieldDescription>
                </Field>
                <Button variant="outline" onClick={clearChat} disabled={!ready}>
                  <Trash2Icon data-icon="inline-start" />
                  Clear chat
                </Button>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                <Field>
                  <FieldLabel htmlFor="tokens">New tokens</FieldLabel>
                  <Input
                    id="tokens"
                    type="number"
                    min={1}
                    max={Math.max(1, contextLimit - 1)}
                    value={maxNewTokens}
                    onChange={(event) => setMaxNewTokens(Number(event.target.value) || 1)}
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="temperature">Temperature</FieldLabel>
                  <Input
                    id="temperature"
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    onChange={(event) => setTemperature(Number(event.target.value))}
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="topK">Top K</FieldLabel>
                  <Input
                    id="topK"
                    type="number"
                    min={1}
                    max={100}
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value) || 1)}
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="repetition">Repetition</FieldLabel>
                  <Input
                    id="repetition"
                    type="number"
                    min={1}
                    max={2}
                    step={0.05}
                    value={repetitionPenalty}
                    onChange={(event) => setRepetitionPenalty(Number(event.target.value) || 1)}
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="ngram">N-gram</FieldLabel>
                  <Input
                    id="ngram"
                    type="number"
                    min={0}
                    max={8}
                    value={noRepeatNgramSize}
                    onChange={(event) => setNoRepeatNgramSize(Number(event.target.value) || 0)}
                  />
                </Field>
              </div>

              {isQwen ? (
                <Field orientation="horizontal">
                  <FieldLabel htmlFor="thinking">Thinking</FieldLabel>
                  <Switch
                    id="thinking"
                    checked={enableThinking}
                    onCheckedChange={(checked) => setEnableThinking(checked === true)}
                  />
                  <FieldDescription>Off injects a closed think block so Qwen answers directly.</FieldDescription>
                </Field>
              ) : null}
            </FieldGroup>
          </CardContent>
        </Card>

        {error ? (
          <Alert variant="destructive">
            <AlertTitle>Could not start the engine</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        <Card className="flex min-h-[420px] flex-1 flex-col">
          <CardHeader className="flex-row items-center justify-between">
            <div className="flex flex-col gap-1">
              <CardTitle>Conversation</CardTitle>
              <CardDescription>{status}</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {tokenRate ? <Badge variant="secondary">{tokenRate}</Badge> : null}
              {architecture ? <Badge variant="outline">{architecture}</Badge> : <Skeleton className="h-5 w-16" />}
            </div>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
            <ScrollArea className="min-h-[240px] flex-1 rounded-lg border p-4">
              {history.length === 0 && assistantDraft === '' ? (
                <Empty>
                  <EmptyHeader>
                    <EmptyMedia variant="icon">
                      <MessageSquareIcon />
                    </EmptyMedia>
                    <EmptyTitle>Send a message to start</EmptyTitle>
                    <EmptyDescription>
                      {ready
                        ? `Loaded ${model.name}. Prompts run entirely in this browser on WebGPU.`
                        : `Loading ${model.name}…`}
                    </EmptyDescription>
                  </EmptyHeader>
                </Empty>
              ) : (
                <div className="flex flex-col gap-3">
                  {history.map((turn, index) => (
                    <ChatBubble key={`${turn.role}-${index}`} turn={turn} />
                  ))}
                  {assistantDraft !== '' ? (
                    <ChatBubble turn={{ role: 'assistant', text: assistantDraft }} streaming />
                  ) : null}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </ScrollArea>

            <form
              onSubmit={(event) => {
                event.preventDefault();
                void sendMessage();
              }}
            >
              <InputGroup>
                <InputGroupTextarea
                  name="message"
                  placeholder={model.prompt}
                  rows={3}
                  value={draft}
                  enterKeyHint="send"
                  disabled={!ready || generating}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.nativeEvent.isComposing || event.keyCode === 229 || event.repeat) {
                      return;
                    }

                    const isEnter = event.key === 'Enter' || event.code === 'Enter' || event.code === 'NumpadEnter';
                    if (
                      isEnter &&
                      event.shiftKey === false &&
                      event.altKey === false &&
                      event.metaKey === false &&
                      event.ctrlKey === false
                    ) {
                      event.preventDefault();
                      void sendMessage();
                    }
                  }}
                />
                <InputGroupAddon align="block-end">
                  {generating ? (
                    <InputGroupButton
                      type="button"
                      variant="destructive"
                      onClick={() => abortRef.current?.abort()}
                      aria-label="Stop generation"
                    >
                      <SquareIcon data-icon="inline-start" />
                      Stop
                    </InputGroupButton>
                  ) : (
                    <InputGroupButton
                      type="submit"
                      variant="default"
                      disabled={!ready || draft.trim() === ''}
                      aria-label="Send message"
                    >
                      {ready ? <ArrowUpIcon data-icon="inline-start" /> : <Spinner data-icon="inline-start" />}
                      {ready ? 'Send' : 'Loading'}
                    </InputGroupButton>
                  )}
                </InputGroupAddon>
              </InputGroup>
            </form>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
