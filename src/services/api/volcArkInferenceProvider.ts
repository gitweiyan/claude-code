/**
 * Built-in inference adapter: Volcengine Ark "Responses API" ↔ Anthropic Messages API shape.
 * Intercepts SDK POST /v1/messages (streaming or not) and forwards to Ark; no external proxy.
 *
 * Env:
 * - CLAUDE_CODE_USE_VOLC_ARK=1 — enable this adapter
 * - ARK_API_URL — default https://ark.cn-beijing.volces.com/api/v3/responses
 * - ARK_API_KEY — optional if x-api-key is already set on the Anthropic request (e.g. ANTHROPIC_API_KEY)
 *
 * Multimodal: user messages with Anthropic `image` blocks (url or base64 data URL)
 * map to Ark `input_image` + `input_text`. `source.type === "file"` is not supported.
 */

import { randomUUID } from 'crypto'
import { APIError } from '@anthropic-ai/sdk'
import { logForDebugging } from '../../utils/debug.js'

const DEFAULT_ARK_URL = 'https://ark.cn-beijing.volces.com/api/v3/responses'

function getArkUrl(): string {
  return (process.env.ARK_API_URL || DEFAULT_ARK_URL).replace(/\/$/, '')
}

function getBearerFromRequest(headers: Headers): string {
  return (
    process.env.ARK_API_KEY ||
    headers.get('Authorization')?.replace(/^Bearer\s+/i, '') ||
    headers.get('x-api-key') ||
    ''
  )
}

/** Ark Responses API user content: plain string or multimodal parts. */
type ArkInputPart =
  | { type: 'input_text'; text: string }
  | { type: 'input_image'; image_url: string }

type ArkUserContent = string | ArkInputPart[]

type ArkInputMessage = { role: string; content: ArkUserContent }

/** Assistant / system: string only (no multimodal in our adapter). */
function flattenAnthropicContent(content: unknown): string {
  if (content === null || content === undefined) return ''
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return String(content)
  const parts: string[] = []
  for (const block of content) {
    if (!block || typeof block !== 'object') continue
    const b = block as Record<string, unknown>
    const t = b.type
    if (t === 'text' && typeof b.text === 'string') parts.push(b.text)
    else if (t === 'tool_use' && typeof b.name === 'string')
      parts.push(`[tool_use:${b.name}]`)
    else if (t === 'tool_result') parts.push('[tool_result]')
    else if (t === 'image') parts.push('[image]')
    else if (t === 'document') parts.push('[document]')
    else parts.push(`[${String(t)}]`)
  }
  return parts.join('\n')
}

function imageUrlFromAnthropicImageBlock(
  block: Record<string, unknown>,
): string | null {
  const src = block.source
  if (!src || typeof src !== 'object') return null
  const s = src as Record<string, unknown>
  if (s.type === 'url' && typeof s.url === 'string' && s.url.length > 0) {
    return s.url
  }
  if (
    s.type === 'base64' &&
    typeof s.data === 'string' &&
    typeof s.media_type === 'string'
  ) {
    return `data:${s.media_type};base64,${s.data}`
  }
  return null
}

/**
 * User message → Ark `input` item: string if text-only, else
 * `[{ type: input_image, image_url }, { type: input_text, text }, …]` per official shape.
 */
function anthropicUserContentToArk(content: unknown): ArkUserContent {
  if (content === null || content === undefined) return ''
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return String(content)

  const parts: ArkInputPart[] = []
  for (const block of content) {
    if (!block || typeof block !== 'object') continue
    const b = block as Record<string, unknown>
    if (b.type === 'text' && typeof b.text === 'string' && b.text.length > 0) {
      parts.push({ type: 'input_text', text: b.text })
    } else if (b.type === 'image') {
      const url = imageUrlFromAnthropicImageBlock(b)
      if (url) parts.push({ type: 'input_image', image_url: url })
      else
        parts.push({ type: 'input_text', text: '[image: unsupported source]' })
    } else if (b.type === 'document') {
      parts.push({ type: 'input_text', text: '[document]' })
    } else if (b.type === 'tool_use' && typeof b.name === 'string') {
      parts.push({ type: 'input_text', text: `[tool_use:${b.name}]` })
    } else if (b.type === 'tool_result') {
      parts.push({ type: 'input_text', text: '[tool_result]' })
    } else if (typeof b.type === 'string') {
      parts.push({ type: 'input_text', text: `[${b.type}]` })
    }
  }

  const merged: ArkInputPart[] = []
  for (const p of parts) {
    const last = merged[merged.length - 1]
    if (
      p.type === 'input_text' &&
      last?.type === 'input_text' &&
      p.text.length > 0
    ) {
      last.text = `${last.text}\n${p.text}`
    } else if (p.type === 'input_text' && p.text.length === 0) {
      continue
    } else {
      merged.push(p)
    }
  }

  if (merged.length === 0) return ''
  if (merged.length === 1 && merged[0].type === 'input_text') {
    return merged[0].text
  }
  return merged
}

function anthropicMessagesToArkInput(
  messages: unknown,
  system: unknown,
): ArkInputMessage[] {
  const out: ArkInputMessage[] = []
  if (system !== undefined && system !== null) {
    if (typeof system === 'string' && system.length > 0) {
      out.push({ role: 'system', content: system })
    } else if (Array.isArray(system)) {
      const sysText = system
        .map(s => {
          if (!s || typeof s !== 'object') return ''
          const o = s as Record<string, unknown>
          if (o.type === 'text' && typeof o.text === 'string') return o.text
          return ''
        })
        .filter(Boolean)
        .join('\n')
      if (sysText) out.push({ role: 'system', content: sysText })
    }
  }
  if (!Array.isArray(messages)) return out
  for (const m of messages) {
    if (!m || typeof m !== 'object') continue
    const msg = m as Record<string, unknown>
    const role = msg.role
    if (role !== 'user' && role !== 'assistant') continue
    if (role === 'user') {
      out.push({
        role: 'user',
        content: anthropicUserContentToArk(msg.content),
      })
    } else {
      out.push({
        role: 'assistant',
        content: flattenAnthropicContent(msg.content),
      })
    }
  }
  return out
}

function buildArkBody(
  anthropicBody: Record<string, unknown>,
): Record<string, unknown> {
  const input = anthropicMessagesToArkInput(
    anthropicBody.messages,
    anthropicBody.system,
  )
  const stream = anthropicBody.stream === true
  const body: Record<string, unknown> = {
    model: anthropicBody.model,
    input,
    stream,
  }
  if (typeof anthropicBody.max_tokens === 'number') {
    body.max_output_tokens = anthropicBody.max_tokens
  }
  return body
}

/** Text blocks in Ark message: output_text (Responses API) or text. */
function textFromArkMessageContent(content: unknown): string {
  if (!Array.isArray(content)) return ''
  const parts: string[] = []
  for (const block of content) {
    if (!block || typeof block !== 'object') continue
    const b = block as Record<string, unknown>
    const t = b.type
    if ((t === 'output_text' || t === 'text') && typeof b.text === 'string') {
      parts.push(b.text)
    }
  }
  return parts.join('')
}

/**
 * Ark Responses API: `output` may include reasoning first, then assistant message.
 * Use the last `type: "message"` item (assistant reply).
 */
function extractArkTextFromJson(obj: unknown): string | null {
  if (obj === null || typeof obj !== 'object') return null
  const o = obj as Record<string, unknown>
  // Chat-completions style
  const choices = o.choices
  if (Array.isArray(choices) && choices[0] && typeof choices[0] === 'object') {
    const c0 = choices[0] as Record<string, unknown>
    const msg = c0.message as Record<string, unknown> | undefined
    if (msg && typeof msg.content === 'string') return msg.content
    const delta = c0.delta as Record<string, unknown> | undefined
    if (delta && typeof delta.content === 'string') return delta.content
  }
  const output = o.output
  if (Array.isArray(output)) {
    for (let i = output.length - 1; i >= 0; i--) {
      const item = output[i]
      if (!item || typeof item !== 'object') continue
      const block = item as Record<string, unknown>
      if (block.type !== 'message') continue
      const text = textFromArkMessageContent(block.content)
      if (text.length > 0) return text
    }
  }
  if (typeof o.text === 'string') return o.text
  return null
}

function extractArkUsageFromJson(obj: unknown): {
  input_tokens: number
  output_tokens: number
} | null {
  if (obj === null || typeof obj !== 'object') return null
  const u = (obj as Record<string, unknown>).usage
  if (!u || typeof u !== 'object') return null
  const usage = u as Record<string, unknown>
  const input = usage.input_tokens
  const output = usage.output_tokens
  if (typeof input !== 'number' || typeof output !== 'number') return null
  return { input_tokens: input, output_tokens: output }
}

function buildAnthropicMessageJson(
  text: string,
  model: string,
  usage?: { input_tokens: number; output_tokens: number } | null,
): string {
  const id = `msg_${randomUUID()}`
  const inTok = usage?.input_tokens ?? 0
  const outTok = usage?.output_tokens ?? 0
  const payload = {
    id,
    type: 'message',
    role: 'assistant',
    content: [{ type: 'text', text }],
    model,
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: inTok,
      output_tokens: outTok,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
  }
  return JSON.stringify(payload)
}

function sseLinePayloads(chunk: string, carry: { buf: string }): string[] {
  carry.buf += chunk
  const lines = carry.buf.split('\n')
  carry.buf = lines.pop() ?? ''
  const payloads: string[] = []
  for (const line of lines) {
    const trimmed = line.replace(/\r$/, '')
    if (trimmed.startsWith('data:')) {
      const data = trimmed.slice(5).trim()
      if (data && data !== '[DONE]') payloads.push(data)
    }
  }
  return payloads
}

/**
 * Best-effort delta extraction from Ark / OpenAI-style stream JSON lines.
 */
function extractStreamDelta(obj: unknown): string {
  const t = extractArkTextFromJson(obj)
  if (t) return t
  if (obj && typeof obj === 'object') {
    const o = obj as Record<string, unknown>
    const delta = o.delta
    if (delta && typeof delta === 'object') {
      const d = delta as Record<string, unknown>
      if (typeof d.text === 'string') return d.text
      const dOut = d.output
      if (Array.isArray(dOut)) {
        let acc = ''
        for (const item of dOut) {
          if (!item || typeof item !== 'object') continue
          const block = item as Record<string, unknown>
          if (block.type === 'message') {
            acc += textFromArkMessageContent(block.content)
          }
        }
        if (acc) return acc
      }
    }
  }
  return ''
}

function encodeSseEvent(event: string, dataObj: unknown): Uint8Array {
  const data = JSON.stringify(dataObj)
  const text = `event: ${event}\ndata: ${data}\n\n`
  return new TextEncoder().encode(text)
}

export function isVolcArkInferenceEnabled(): boolean {
  const v = process.env.CLAUDE_CODE_USE_VOLC_ARK
  if (!v) return false
  const n = v.toLowerCase().trim()
  return n === '1' || n === 'true' || n === 'yes' || n === 'on'
}

/**
 * Wrap fetch: only intercepts Anthropic Messages POST; other requests pass through.
 */
export function createVolcArkInferenceFetch(
  innerFetch: typeof fetch,
): typeof fetch {
  return async (input, init) => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url
    const method = (init?.method || 'GET').toUpperCase()

    if (method !== 'POST' || !url.includes('/v1/messages')) {
      return innerFetch(input, init)
    }

    let anthropicBody: Record<string, unknown>
    try {
      const raw = init?.body
      if (typeof raw === 'string') {
        anthropicBody = JSON.parse(raw) as Record<string, unknown>
      } else if (raw instanceof Uint8Array) {
        anthropicBody = JSON.parse(new TextDecoder().decode(raw)) as Record<
          string,
          unknown
        >
      } else {
        return innerFetch(input, init)
      }
    } catch {
      return innerFetch(input, init)
    }

    const reqHeaders = new Headers(init?.headers)
    const bearer = getBearerFromRequest(reqHeaders)
    if (!bearer) {
      throw new APIError(
        401,
        undefined,
        'Missing ARK_API_KEY or ANTHROPIC_API_KEY for Volc Ark adapter',
        new Headers(),
      )
    }

    const arkBody = buildArkBody(anthropicBody)
    const model = String(anthropicBody.model ?? '')
    const stream = arkBody.stream === true

    const arkHeaders: Record<string, string> = {
      Authorization: `Bearer ${bearer}`,
      'Content-Type': 'application/json',
      Accept: stream ? 'text/event-stream' : 'application/json',
    }

    const arkUrl = getArkUrl()
    logForDebugging(
      `[volc-ark] ${stream ? 'stream' : 'request'} model=${model} -> ${arkUrl}`,
    )

    const arkRes = await innerFetch(arkUrl, {
      method: 'POST',
      headers: arkHeaders,
      body: JSON.stringify(arkBody),
      signal: init?.signal,
    })

    if (!arkRes.ok) {
      const errText = await arkRes.text()
      let message = errText
      try {
        const j = JSON.parse(errText) as Record<string, unknown>
        const err = j.error as Record<string, unknown> | undefined
        if (err && typeof err.message === 'string') message = err.message
      } catch {
        /* keep text */
      }
      throw new APIError(arkRes.status, undefined, message, arkRes.headers)
    }

    if (!stream) {
      const text = await arkRes.text()
      let data: unknown
      try {
        data = JSON.parse(text) as unknown
      } catch {
        throw new APIError(
          500,
          undefined,
          'Ark returned non-JSON response',
          new Headers(),
        )
      }
      const outText = extractArkTextFromJson(data) ?? ''
      const usage = extractArkUsageFromJson(data)
      const anthropicJson = buildAnthropicMessageJson(outText, model, usage)
      return new Response(anthropicJson, {
        status: 200,
        headers: {
          'content-type': 'application/json',
        },
      })
    }

    // Streaming: convert Ark SSE → Anthropic SSE events
    const msgId = `msg_${randomUUID()}`
    const inBody = arkRes.body
    if (!inBody) {
      throw new APIError(
        500,
        undefined,
        'Ark stream has empty body',
        new Headers(),
      )
    }

    const reader = inBody.getReader()
    const decoder = new TextDecoder()
    const carry = { buf: '' }
    let started = false
    let bufText = ''

    const outStream = new ReadableStream<Uint8Array>({
      async start(controller) {
        const enqueueStart = () => {
          controller.enqueue(
            encodeSseEvent('message_start', {
              type: 'message_start',
              message: {
                id: msgId,
                type: 'message',
                role: 'assistant',
                content: [],
                model,
                stop_reason: null,
                stop_sequence: null,
                usage: {
                  input_tokens: 0,
                  output_tokens: 0,
                  cache_creation_input_tokens: 0,
                  cache_read_input_tokens: 0,
                },
              },
            }),
          )
          controller.enqueue(
            encodeSseEvent('content_block_start', {
              type: 'content_block_start',
              index: 0,
              content_block: { type: 'text', text: '' },
            }),
          )
        }

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            const chunk = decoder.decode(value, { stream: true })
            const payloads = sseLinePayloads(chunk, carry)
            for (const payload of payloads) {
              let obj: unknown
              try {
                obj = JSON.parse(payload) as unknown
              } catch {
                continue
              }
              const delta = extractStreamDelta(obj)
              if (delta) {
                if (!started) {
                  started = true
                  enqueueStart()
                }
                bufText += delta
                controller.enqueue(
                  encodeSseEvent('content_block_delta', {
                    type: 'content_block_delta',
                    index: 0,
                    delta: { type: 'text_delta', text: delta },
                  }),
                )
              }
            }
          }
          const tailPayloads = sseLinePayloads('\n', carry)
          for (const payload of tailPayloads) {
            let obj: unknown
            try {
              obj = JSON.parse(payload) as unknown
            } catch {
              continue
            }
            const delta = extractStreamDelta(obj)
            if (delta) {
              if (!started) {
                started = true
                enqueueStart()
              }
              bufText += delta
              controller.enqueue(
                encodeSseEvent('content_block_delta', {
                  type: 'content_block_delta',
                  index: 0,
                  delta: { type: 'text_delta', text: delta },
                }),
              )
            }
          }

          if (!started && bufText.length === 0) {
            // No SSE deltas; try full body as JSON (some gateways buffer)
            const full = carry.buf.trim()
            if (full) {
              try {
                const obj = JSON.parse(full) as unknown
                const t = extractArkTextFromJson(obj)
                if (t) {
                  started = true
                  enqueueStart()
                  bufText = t
                  controller.enqueue(
                    encodeSseEvent('content_block_delta', {
                      type: 'content_block_delta',
                      index: 0,
                      delta: { type: 'text_delta', text: t },
                    }),
                  )
                }
              } catch {
                /* ignore */
              }
            }
          }

          if (!started) {
            controller.enqueue(
              encodeSseEvent('message_start', {
                type: 'message_start',
                message: {
                  id: msgId,
                  type: 'message',
                  role: 'assistant',
                  content: [],
                  model,
                  stop_reason: null,
                  stop_sequence: null,
                  usage: {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                  },
                },
              }),
            )
            controller.enqueue(
              encodeSseEvent('content_block_start', {
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '' },
              }),
            )
          }

          controller.enqueue(
            encodeSseEvent('content_block_stop', {
              type: 'content_block_stop',
              index: 0,
            }),
          )
          const outTok = Math.max(0, Math.ceil(bufText.length / 4))
          controller.enqueue(
            encodeSseEvent('message_delta', {
              type: 'message_delta',
              delta: { stop_reason: 'end_turn', stop_sequence: null },
              usage: {
                input_tokens: 0,
                output_tokens: outTok,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
              },
            }),
          )
          controller.enqueue(
            encodeSseEvent('message_stop', { type: 'message_stop' }),
          )
        } catch (e) {
          controller.error(e)
          return
        } finally {
          controller.close()
        }
      },
    })

    return new Response(outStream, {
      status: 200,
      headers: {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      },
    })
  }
}
