import { completionFollowUpText, formatPrompt } from 'three-llm';
import { DEFAULT_MODEL_ID, MODEL_CATALOG, resolveModelURL } from 'three-llm/catalog';
import type { ChatMessage, GenerateOptions, GenerationResult, ModelCatalogEntry } from 'three-llm';
import { ArrowUpIcon, GithubIcon, MessageSquareIcon, SettingsIcon, SquareIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';

import { MarkdownContent } from '@/components/MarkdownContent';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Empty, EmptyHeader, EmptyTitle } from '@/components/ui/empty';
import { Field, FieldDescription, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea } from '@/components/ui/input-group';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Spinner } from '@/components/ui/spinner';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

type TslRunner = {
  maxTokens: number;
  generate: (renderer: unknown, prompt: string, options: GenerateOptions) => Promise<GenerationResult>;
  prepare?: (renderer: unknown) => void;
  resetCache: () => void;
  weights: {
    architecture: string;
    tokenizer: { decode: (ids: number[]) => string; encode: (text: string) => number[] };
    formatChat?: (messages: ChatMessage[], options?: { enableThinking?: boolean }) => string;
  };
};

type ChatTurn = { role: 'user' | 'assistant'; text: string };

function NpmIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden {...props}>
      <path d="M0 0v24h24V0H0zm19.2 19.2h-4.8V8.4H9.6v10.8H4.8V4.8h14.4v14.4z" />
    </svg>
  );
}

const DEFAULT_MAX_NEW_TOKENS = 1024;
const MOBILE_MAX_TOKENS = 1024;
const DRAFT_FLUSH_MS = 1000;

function isConstrainedDevice() {
  if (typeof navigator === 'undefined') return false;
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { mobile?: boolean } };
  if (nav.userAgentData?.mobile === true) return true;
  if (nav.deviceMemory !== undefined && nav.deviceMemory <= 4) return true;
  return /Mobi|Android|iPhone|iPad/i.test(nav.userAgent);
}

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
  return formatPrompt(runner.weights, turns, { enableThinking });
}

export function ChatApp({ modelId, onModelChange }: { modelId?: string; onModelChange: (id: string) => void }) {
  const model = useMemo(() => selectedModel(modelId), [modelId]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const runnerRef = useRef<TslRunner | undefined>(undefined);
  const rendererRef = useRef<{ dispose: () => void } | undefined>(undefined);
  const conversationTokensRef = useRef<number[] | undefined>(undefined);
  const abortRef = useRef<AbortController | undefined>(undefined);
  const generationIdRef = useRef(0);
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

  const isQwen = (architecture ?? model.id).includes('qwen');
  const busy = !ready || generating;
  const canClear = ready && (history.length > 0 || assistantDraft !== '');

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
          maxTokens: isConstrainedDevice() ? MOBILE_MAX_TOKENS : undefined,
          onProgress: (message) => {
            if (!cancelled) setStatus(message);
          },
        })) as TslRunner;

        if (cancelled) {
          runnerRef.current = undefined;
          renderer.dispose();
          return;
        }

        runner.prepare?.(renderer);

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
    generationIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = undefined;
    setGenerating(false);
    setHistory([]);
    resetAssistantDraft();
    conversationTokensRef.current = undefined;
    runnerRef.current?.resetCache();
    if (runnerRef.current) setStatus(`Ready. Context ${runnerRef.current.maxTokens} tokens.`);
  }

  function changeModel(nextId: string) {
    if (nextId === model.id) return;
    onModelChange(nextId);
  }

  async function sendMessage() {
    const runner = runnerRef.current;
    const renderer = rendererRef.current;
    const userText = draft.trim();
    if (!userText || !runner || !renderer || generating) return;

    const abortController = new AbortController();
    abortRef.current = abortController;
    const generationId = generationIdRef.current + 1;
    generationIdRef.current = generationId;
    setGenerating(true);
    setDraft('');
    resetAssistantDraft();

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
          if (generationId !== generationIdRef.current) return;
          const fresh = promptTokens - cachedPromptTokens;
          setStatus(
            cachedPromptTokens > 0
              ? `Prefilling ${fresh} new prompt token(s); ${cachedPromptTokens} cached…`
              : `Prefilling ${fresh} prompt token(s)…`,
          );
        },
        onPrefillProgress: ({ cachedPromptTokens, completedPromptTokens = 0, promptTokens }) => {
          if (generationId !== generationIdRef.current) return;
          const fresh = promptTokens - cachedPromptTokens;
          const completed = Math.max(0, completedPromptTokens - cachedPromptTokens);
          setStatus(
            cachedPromptTokens > 0
              ? `Prefilling ${Math.min(completed, fresh)}/${fresh} new prompt token(s); ${cachedPromptTokens} cached…`
              : `Prefilling ${completedPromptTokens}/${promptTokens} prompt token(s)…`,
          );
        },
        onToken: (_text, nextToken) => {
          if (generationId !== generationIdRef.current) return;
          generatedIds.push(nextToken);
          queueAssistantDraft(runner.weights.tokenizer.decode(generatedIds));
          const elapsed = (performance.now() - generationStart) / 1000;
          if (generatedIds.length > 0 && elapsed > 0) {
            setStatus(`Generating… ${(generatedIds.length / elapsed).toFixed(1)} tok/s`);
          }
        },
      };

      if (conversationTokensRef.current && conversationTokensRef.current.length > 0 && !runner.weights.formatChat) {
        generateOptions.inputTokens = conversationTokensRef.current.concat(
          runner.weights.tokenizer.encode(completionFollowUpText(userText)),
        );
      }

      const result = await runner.generate(
        renderer,
        conversationPrompt(runner, nextHistory, enableThinking),
        generateOptions,
      );
      if (generationId !== generationIdRef.current) return;

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
      setStatus(
        result.aborted
          ? `Stopped after ${result.generatedTokens.length} token(s).${rate}${cacheNote}`
          : `Generated ${result.generatedTokens.length} token(s).${rate}${cacheNote}`,
      );
    } catch (generateError) {
      if (generationId !== generationIdRef.current) return;
      const message = generateError instanceof Error ? generateError.message : String(generateError);
      setStatus(message);
      toast.error('Generation failed', { description: message });
      if (assistantDraftLatestRef.current !== '') {
        setHistory([...nextHistory, { role: 'assistant', text: assistantDraftLatestRef.current }]);
      }
      resetAssistantDraft();
    } finally {
      if (generationId === generationIdRef.current) {
        abortRef.current = undefined;
        setGenerating(false);
      }
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <canvas ref={canvasRef} className="hidden" width={1} height={1} aria-hidden />
      <header className="border-b bg-background">
        <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex flex-col gap-0.5">
            <p className="text-sm font-medium">three-llm</p>
            <p className="text-muted-foreground text-sm">WebGPU LLM chat in the browser</p>
          </div>
          <div className="flex items-center gap-1">
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

      <main className="mx-auto flex w-full max-w-3xl min-h-0 flex-1 flex-col px-4">
        {error ? (
          <Alert variant="destructive" className="mt-4">
            <AlertTitle>Could not start the engine</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        <div className="relative min-h-0 flex-1">
          {canClear ? (
            <div className="pointer-events-none absolute inset-y-0 right-0 z-10 flex items-start pt-3">
              <Button
                variant="outline"
                size="sm"
                className="pointer-events-auto"
                onClick={clearChat}
                aria-label="New conversation"
              >
                New
                <MessageSquareIcon data-icon="inline-end" />
              </Button>
            </div>
          ) : null}
          <ScrollArea className="h-full">
            {history.length === 0 && assistantDraft === '' ? (
              <Empty className="h-full min-h-64">
                <EmptyHeader>
                  <EmptyTitle>{ready ? 'Send a message to start the conversation.' : 'Loading model...'}</EmptyTitle>
                </EmptyHeader>
              </Empty>
            ) : (
              <div className="flex flex-col gap-3 py-6 pr-24">
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
        </div>

        <div className="bg-background flex flex-col gap-2 pt-2 pb-4">
          <p className="text-muted-foreground flex items-center gap-2 text-sm" aria-live="polite">
            {busy && error === null ? <Spinner /> : null}
            <span>{status}</span>
          </p>

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
              <InputGroupAddon align="block-end" className="justify-end">
                <div className="flex items-center gap-1">
                  <Select
                    value={model.id}
                    onValueChange={(value) => {
                      if (typeof value === 'string') changeModel(value);
                    }}
                    disabled={generating}
                  >
                    <SelectTrigger size="sm" type="button" aria-label="Model">
                      <SelectValue>{model.name}</SelectValue>
                    </SelectTrigger>
                    <SelectContent side="top" align="end" alignItemWithTrigger={false} className="min-w-64">
                      <SelectGroup>
                        {MODEL_CATALOG.map((entry) => (
                          <SelectItem key={entry.id} value={entry.id}>
                            <span className="flex items-center gap-2">
                              {entry.name}
                              <span className="text-muted-foreground">{entry.sizeHint}</span>
                              {entry.badge ? <Badge variant="secondary">{entry.badge}</Badge> : null}
                            </span>
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>

                  <Dialog>
                    <DialogTrigger
                      render={
                        <InputGroupButton
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          aria-label="Generation settings"
                        />
                      }
                    >
                      <SettingsIcon />
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle>Generation settings</DialogTitle>
                        <DialogDescription>{model.note}</DialogDescription>
                      </DialogHeader>
                      <FieldGroup>
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
                        {isQwen ? (
                          <Field orientation="horizontal">
                            <FieldLabel htmlFor="thinking">Thinking</FieldLabel>
                            <Switch
                              id="thinking"
                              checked={enableThinking}
                              onCheckedChange={(checked) => setEnableThinking(checked === true)}
                            />
                            <FieldDescription>
                              Off injects a closed think block so Qwen answers directly.
                            </FieldDescription>
                          </Field>
                        ) : null}
                      </FieldGroup>
                      <DialogFooter showCloseButton />
                    </DialogContent>
                  </Dialog>
                </div>

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
        </div>
      </main>
    </div>
  );
}
