// werewolf-agent-01's house baseline, and the starter every kit carries (server/competitions/kit-dirs.ts
// KIT_STARTER_DIR): a frontier model, asked the plain way. Every decision the env waits on goes to MODEL
// once, as the game's rules, that one game's view and the decision, with a request for one JSON object: no
// tools, no strategy coaching, no memory across decisions or games, no second opinion, no voting. MODEL is
// claude-opus-5-5 thinking at EFFORT low. A prize needs a mean score above this baseline's
// (competition.yaml prize_requires_beating_baseline).
//
// The run: POST /observe, and every decision it lists that is not already being answered is started, at most
// CONCURRENCY at a time, in the order the env lists them. For each: one Anthropic Messages call through the
// gateway (ANTHROPIC_BASE_URL/v1/messages), the reply's last JSON object read as the answer, and POST /act.
// Then /observe again: when every decision it lists is already in flight the run waits for one to finish
// first, and when none waits the env holds the request open until one does. The run exits once "done".
//
//   the prompt     the system prompt is the brief's rules (its "## The game" and "## The score" sections,
//                  read from MAKEFUN_PROMPT_FILE); the user message is the game's view as JSON, as /observe
//                  served it, and the decision: its kind, its options or its length limit, and the JSON shape
//                  of the answer. Nothing names another game or the trial's progress
//   caching        the system prompt is one text block with a cache_control breakpoint (ephemeral, 5 minutes):
//                  it is the same in every call, so the calls after the first read it from the cache, and the
//                  user message after it, different in every call, is never written to the cache
//   thinking       sent as `thinking: {type: 'adaptive'}` with `output_config: {effort: EFFORT}` (Claude 5.5
//                  takes no thinking token budget); should the model answer that request 400
//                  (invalid_request_error), that decision is asked again without either
//   the answer     speak: {"say": "<statement>"} or {"say": null}, the statement folded to one line the env's
//                  way and cut to the decision's max_chars; a target decision: {"target": <seat>} or
//                  {"target": null}, a seat outside the options read as null. A reply with no such object,
//                  and an answer the env refuses, falls back to a pass or an abstention, so no game stalls
//   an upstream    a 429, a 5xx or a 529 is asked once more, after the wait its Retry-After names
//   error          (delay-seconds, at most RETRY_AFTER_MAX_SEC) or RETRY_WAIT_SEC when it names none; a second
//                  error, a gateway refusal (budget_exhausted, request_cap) or a call that throws falls back
//   in flight      at most CONCURRENCY decisions at once, the env's own cap on live games: one decision per
//                  live game waits at a time, so no game ever waits on another's reply
//
// Deterministic given the model's answers (the replay serves the recorded ones): no clock, no randomness,
// and every request a function of its own game's view and the replies to its own earlier tries alone, so
// whatever order the replies come back in, the replay sends the same requests. A wait changes when a
// request goes, never what it is. Run by the grader as `node /bot/main.ts` with this directory at /bot;
// Node 22's standard library only.
//
// Whose key pays: entered as your own entry, every call runs on the Anthropic key you push to the competition
// before you submit (ANTHROPIC_API_KEY=<your key> in a .env file, then `makefun secrets push
// werewolf-agent-01 --env-file .env`, or Profile > API keys); with none pushed every call is answered 401
// and every decision falls back. Inside the trial ANTHROPIC_API_KEY holds only a dummy the gateway swaps for
// your key, which never enters the sandbox; the house's own baseline run is paid by the house.

import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

export const MODEL = 'claude-opus-5-5';
export const EFFORT = 'low';
/** max_tokens of every call: the entry lane's max_output_tokens_per_request. Thinking counts toward it. */
export const MAX_TOKENS = 2048;
export const ANTHROPIC_VERSION = '2023-06-01';
/** The most decisions in flight at once: the env's live games, each waiting on at most one decision. */
export const CONCURRENCY = 10;
/** The wait before an upstream error's one retry when its answer names no Retry-After, in seconds. */
export const RETRY_WAIT_SEC = 5;
/** The longest Retry-After waited for, in seconds. */
export const RETRY_AFTER_MAX_SEC = 20;
/** The entry lane's max_request_bytes: a request over it has its oldest statements clipped until it fits. */
export const MAX_REQUEST_BYTES = 131_072;
/** A clipped statement keeps this many characters, then "…". */
export const CLIPPED_CHARS = 80;
/** The brief's sections the system prompt carries. */
export const RULE_SECTIONS = ['The game', 'The score'] as const;

type Seat = number;
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  text?: string;
  [k: string]: unknown;
}

interface View {
  day: number;
  phase: string;
  you: { seat: Seat; name: string; role: string; alive: boolean; partner: { seat: Seat; name: string; alive: boolean } | null };
  log: LogEntry[];
  [k: string]: unknown;
}

/** One decision waiting on the entry, as /observe lists it. */
export interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

/** The brief's rule sections (RULE_SECTIONS, heading included, in the brief's order); the whole brief when it has none of them. */
export function rulesFrom(brief: string): string {
  const parts = brief.split(/^(?=## )/m);
  const kept = parts.filter((p) => RULE_SECTIONS.some((s) => p.startsWith(`## ${s}\n`))).map((p) => p.trim());
  return kept.length > 0 ? kept.join('\n\n') : brief.trim();
}

/** The system prompt: the brief's rules, and whose seat the model plays. */
export function systemFor(rules: string): string {
  return [
    'You are an expert Werewolf player. Play to win, not to explain the rules.',
    'Return exactly one valid JSON object matching the requested schema. Never include markdown or extra JSON.',
    'Use only the current game view. Track revealed roles, deaths, votes, claims, contradictions and voting alliances.',
    'Village: prioritize confirmed information and mechanically plausible wolf pairs; do not overtrust confident claims. Vote a living suspected wolf, never yourself.',
    'Seer: inspect the most informative living seat; publicly report a confirmed wolf and vote them, or defend a confirmed villager. If counterclaimed, state concrete night results.',
    'Doctor: protect the most likely seer or the strongest confirmed village leader; never repeat last protection and never protect yourself.',
    'Werewolf: coordinate with your partner, avoid voting your partner, kill confirmed seers and influential villagers, keep public claims consistent, and vote a credible village target. During debate, sound concise and uncertain rather than omniscient.',
    'Speak with specific evidence from the public log. Do not reveal private knowledge unless it is a deliberate in-character claim.',
    '', 'These are the rules:', '', rules,
  ].join('\n');
}

const seatList = (seats: readonly Seat[]): string => seats.map((s) => `seat ${s}`).join(', ');

/** The decision, as the user message states it after the view: what to answer and the JSON shape to answer in. */
export function decisionText(p: Pending): string {
  const options = p.options ?? [];
  switch (p.decision) {
    case 'speak':
      return [
        `It is your turn to speak in the day's debate. Make one statement of at most ${p.max_chars ?? 500} characters, or pass.`,
        'Reply with one JSON object and nothing after it: {"say": "<your statement>"}, or {"say": null} to pass.',
      ].join('\n');
    case 'vote':
      return [`It is your turn to vote. Vote to exile one of: ${seatList(options)}; or abstain.`, 'Reply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null} to abstain.'].join('\n');
    case 'kill':
      return [`It is night: name the seat you want the werewolves to kill, one of: ${seatList(options)}; or abstain.`, 'Reply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null} to abstain.'].join('\n');
    case 'protect':
      return [`It is night: choose the seat to protect, one of: ${seatList(options)}; or nobody.`, 'Reply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null} for nobody.'].join('\n');
    case 'inspect':
      return [`It is night: choose the seat to inspect, one of: ${seatList(options)}; or nobody.`, 'Reply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null} for nobody.'].join('\n');
  }
}

/** The user message: the game's view as JSON, then the decision. */
export function userText(p: Pending): string {
  return ['Your view of the game (JSON):', JSON.stringify(p.view), '', `Your decision (${p.decision}):`, decisionText(p)].join('\n');
}

/** The request's bytes as it is sent. */
export const requestBytes = (request: Record<string, unknown>): number => Buffer.byteLength(JSON.stringify(request), 'utf8');

function bodyOf(system: string, p: Pending, thinking: boolean): Record<string, unknown> {
  return {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    ...(thinking ? { thinking: { type: 'adaptive' }, output_config: { effort: EFFORT } } : {}),
    // The rules are the same in every call: one cache breakpoint on them, so later calls read them from the cache.
    system: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }],
    messages: [{ role: 'user', content: userText(p) }],
  };
}

/**
 * The Anthropic Messages request for one decision: the rules as the system prompt, the game's view and the
 * decision as the one user message, thinking at EFFORT when `thinking`. Should it exceed MAX_REQUEST_BYTES,
 * the oldest statements of the view's log are clipped to CLIPPED_CHARS, one at a time, until it fits.
 */
export function requestFor(rules: string, p: Pending, thinking: boolean): Record<string, unknown> {
  const system = systemFor(rules);
  let request = bodyOf(system, p, thinking);
  if (requestBytes(request) <= MAX_REQUEST_BYTES) return request;
  const view = structuredClone(p.view);
  const clipped = { ...p, view };
  for (const entry of view.log) {
    if (entry.kind !== 'speech' || typeof entry.text !== 'string' || [...entry.text].length <= CLIPPED_CHARS) continue;
    entry.text = `${[...entry.text].slice(0, CLIPPED_CHARS).join('')}…`;
    request = bodyOf(system, clipped, thinking);
    if (requestBytes(request) <= MAX_REQUEST_BYTES) break;
  }
  return request;
}

/** Every JSON object in a text, in order (a brace inside a string skipped). */
function jsonObjects(text: string): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          try {
            const v: unknown = JSON.parse(text.slice(start, i + 1));
            if (isRecord(v)) out.push(v);
          } catch {
            // Not JSON: look on.
          }
          break;
        }
      }
    }
  }
  return out;
}

/** A statement folded to one line the env's way: runs of line breaks and tabs become one space, other control characters go, the ends are trimmed. */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

/** The pass or abstention for a decision: what an unreadable reply becomes. */
export function fallbackFor(p: Pick<Pending, 'game' | 'decision' | 'options' | 'max_chars' | 'view'>): Record<string, unknown> {
  const options = p.options ?? [];
  const view = p.view as View | undefined;
  const you = view?.you;
  const players = Array.isArray((view as any)?.players) ? (view as any).players as Array<{ seat: number; alive: boolean; role?: string | null }> : [];
  const living = players.filter((x) => x.alive && x.seat !== you?.seat).map((x) => x.seat);
  const revealedWolves = players.filter((x) => x.alive && x.role === 'werewolf' && x.seat !== you?.seat).map((x) => x.seat);
  const publicWolves = revealedWolves.filter((s) => options.includes(s));
  const pick = publicWolves[0] ?? options.filter((s) => s !== you?.seat).sort((a, b) => a - b)[0] ?? null;
  if (p.decision === 'speak') {
    const text = you?.role === 'werewolf'
      ? 'Saya fokus pada urutan vote dan kontradiksi, bukan tuduhan tanpa bukti.'
      : publicWolves.length > 0
        ? `Seat ${publicWolves[0]} sudah terkonfirmasi werewolf; vote seat itu.`
        : 'Saya belum melihat bukti keras. Bandingkan klaim, urutan vote, dan target malam.';
    return text.length <= (p.max_chars ?? 500) ? { game: p.game, decision: 'speak', text } : { game: p.game, decision: 'speak', pass: true };
  }
  if (p.decision === 'protect') {
    const seer = players.find((x) => x.alive && (x.role === 'seer' || x.role === 'villager'))?.seat ?? null;
    return { game: p.game, decision: p.decision, target: options.includes(seer as number) ? seer : (options[0] ?? null) };
  }
  if (p.decision === 'inspect') return { game: p.game, decision: p.decision, target: options[0] ?? null };
  return { game: p.game, decision: p.decision, target: pick };
}

export type Parsed = { ok: true; act: Record<string, unknown> } | { ok: false; act: Record<string, unknown> };

/**
 * The /act body a reply's text answers a decision with: the last JSON object holding "say" (speak) or
 * "target" (a target decision). A statement is folded and cut to max_chars, and one that folds to nothing is
 * a pass; a target that is not one of the options (a seat number written as a string read as the number) is
 * an abstention. With no such object the result is the fallback, marked not ok.
 */
export function parseReply(text: string, p: Pick<Pending, 'game' | 'decision' | 'options' | 'max_chars'>): Parsed {
  const key = p.decision === 'speak' ? 'say' : 'target';
  const found = jsonObjects(text).filter((o) => key in o);
  if (found.length === 0) return { ok: false, act: fallbackFor(p) };
  const value = found[found.length - 1]![key];
  if (p.decision === 'speak') {
    if (typeof value !== 'string') return { ok: value === null, act: fallbackFor(p) };
    const folded = foldStatement(value);
    if (folded === '') return { ok: true, act: fallbackFor(p) };
    const max = p.max_chars ?? 500;
    const cut = [...folded].length > max ? [...folded].slice(0, max).join('').trim() : folded;
    return { ok: true, act: { game: p.game, decision: 'speak', text: cut } };
  }
  const seat = typeof value === 'string' && /^\s*\d+\s*$/.test(value) ? Number(value) : value;
  const target = typeof seat === 'number' && (p.options ?? []).includes(seat) ? seat : null;
  return { ok: value === null || target !== null, act: { game: p.game, decision: p.decision, target } };
}

/** A message's text: its text blocks joined (its thinking blocks left out); '' when it has none. */
export function replyText(body: unknown): string {
  const content = isRecord(body) ? body.content : undefined;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/**
 * How long to wait before an upstream error's one retry, in ms: its Retry-After in delay-seconds, at most
 * RETRY_AFTER_MAX_SEC; RETRY_WAIT_SEC when it names none or names a date (read against no clock).
 */
export function retryWaitMs(retryAfter: string | null): number {
  const v = (retryAfter ?? '').trim();
  if (!/^\d+$/.test(v)) return RETRY_WAIT_SEC * 1000;
  return Math.min(Number(v), RETRY_AFTER_MAX_SEC) * 1000;
}

interface Reply {
  status: number;
  body: unknown;
  /** The answer's Retry-After header, or null. */
  retryAfter: string | null;
}

async function readReply(res: Response): Promise<Reply> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body, retryAfter: res.headers.get('retry-after') };
}

/** The gateway's refusals that spending can cause: final, so the decision is not asked again. */
const SPENT = new Set(['budget_exhausted', 'request_cap']);

/** An error document's type: Anthropic's {"type": "error", "error": {"type"}}, which the gateway's refusals take too; '' when none. */
const errorType = (body: unknown): string => {
  const e = isRecord(body) ? body.error : undefined;
  return isRecord(e) && typeof e.type === 'string' ? e.type : '';
};

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export interface RunOptions {
  envUrl?: string;
  llmUrl?: string;
  apiKey?: string;
  /** The brief's text; read from MAKEFUN_PROMPT_FILE when omitted. */
  brief?: string;
  /** Where the run's summary goes (stdout by default). */
  log?: (line: string) => void;
}

export interface RunSummary {
  decisions: number;
  /** Decisions answered from a readable reply. */
  answered: number;
  /** Decisions that fell back to a pass or an abstention. */
  fallbacks: number;
  calls: number;
  refused: Map<string, number>;
}

export async function run(o: RunOptions = {}): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (o.envUrl ?? env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const llmUrl = (o.llmUrl ?? env.ANTHROPIC_BASE_URL ?? '').replace(/\/+$/, '');
  const apiKey = o.apiKey ?? env.ANTHROPIC_API_KEY ?? '';
  const log = o.log ?? ((line: string) => process.stdout.write(`${line}\n`));
  if (envUrl === '' || llmUrl === '') throw new Error('MAKEFUN_ENV_URL and ANTHROPIC_BASE_URL are both needed');
  let brief = o.brief;
  if (brief === undefined) {
    const file = env.MAKEFUN_PROMPT_FILE ?? '';
    if (file === '') throw new Error('MAKEFUN_PROMPT_FILE is needed');
    brief = readFileSync(file, 'utf8');
  }
  const rules = rulesFrom(brief);
  const summary: RunSummary = { decisions: 0, answered: 0, fallbacks: 0, calls: 0, refused: new Map() };

  const envPost = async (path: string, body: unknown): Promise<Reply> =>
    readReply(await fetch(`${envUrl}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));
  const messages = async (request: Record<string, unknown>): Promise<Reply> => {
    summary.calls += 1;
    return readReply(
      await fetch(`${llmUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'anthropic-version': ANTHROPIC_VERSION, 'x-api-key': apiKey },
        body: JSON.stringify(request),
      }),
    );
  };

  /** The model's reply to one decision, or null when there is none to read (see the header). */
  const ask = async (p: Pending): Promise<Reply | null> => {
    let thinking = true;
    let reply = await messages(requestFor(rules, p, thinking));
    if (reply.status === 400 && errorType(reply.body) === 'invalid_request_error') {
      thinking = false;
      reply = await messages(requestFor(rules, p, thinking));
    }
    if (SPENT.has(errorType(reply.body))) {
      summary.refused.set(errorType(reply.body), (summary.refused.get(errorType(reply.body)) ?? 0) + 1);
      return null;
    }
    if (reply.status === 429 || reply.status >= 500) {
      await sleep(retryWaitMs(reply.retryAfter));
      reply = await messages(requestFor(rules, p, thinking));
      if (SPENT.has(errorType(reply.body))) {
        summary.refused.set(errorType(reply.body), (summary.refused.get(errorType(reply.body)) ?? 0) + 1);
        return null;
      }
    }
    return reply.status === 200 ? reply : null;
  };

  /** One decision: the call, the answer posted, and the fallback posted should the env refuse it. */
  const decide = async (p: Pending): Promise<void> => {
    summary.decisions += 1;
    let act = fallbackFor(p);
    let ok = false;
    try {
      const reply = await ask(p);
      if (reply !== null) ({ ok, act } = parseReply(replyText(reply.body), p));
    } catch {
      // The gateway gone or the call cut off: the decision falls back.
    }
    let posted = await envPost('/act', act);
    if (posted.status === 400 && JSON.stringify(act) !== JSON.stringify(fallbackFor(p))) {
      ok = false;
      posted = await envPost('/act', fallbackFor(p));
    }
    if (ok && posted.status === 200) summary.answered += 1;
    else summary.fallbacks += 1;
  };

  // A decision is known by its game, its kind and how far its game's log has come: the same decision is
  // listed again on every /observe until it is answered, and never started twice.
  const keyOf = (p: Pending): string => `${p.game}:${p.decision}:${p.view.log.length}`;
  const inFlight = new Map<string, Promise<void>>();
  const started = new Set<string>();
  const errors: string[] = [];
  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    for (const p of obs.pending) {
      const key = keyOf(p);
      if (started.has(key) || inFlight.size >= CONCURRENCY) continue;
      started.add(key);
      const task = decide(p)
        .catch((err: unknown) => {
          errors.push(`game ${p.game}: ${(err as Error).message}`);
        })
        .finally(() => inFlight.delete(key));
      inFlight.set(key, task);
    }
    const waiting = obs.pending.some((p) => !started.has(keyOf(p)));
    if (inFlight.size > 0 && (inFlight.size >= CONCURRENCY || !waiting)) await Promise.race(inFlight.values());
    else if (inFlight.size === 0 && obs.pending.length > 0 && !waiting) await sleep(250); // listed, answered, not yet moved on
    // Only the env can fail a decision (a model call that throws falls back): the env gone ends the run.
    if (errors.length > 0) break;
  }
  await Promise.all(inFlight.values());
  if (errors.length > 0) throw new Error(`the env failed: ${errors.join('; ')}`);
  const spent = [...summary.refused.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([kind, n]) => `${n} refused by the gateway (${kind})`);
  log(`baseline: ${summary.decisions} decisions, ${summary.answered} answered by the model, ${summary.fallbacks} fell back; ${summary.calls} model calls${spent.length > 0 ? `; ${spent.join(', ')}` : ''}`);
  return summary;
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`baseline: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
