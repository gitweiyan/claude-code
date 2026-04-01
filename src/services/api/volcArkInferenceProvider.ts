/**
 * Built-in inference adapter: Volcengine Ark "Responses API" ↔ Anthropic Messages API shape.
 * Intercepts SDK POST /v1/messages (streaming or not) and forwards to Ark; no external proxy.
 *
 * Env:
 * - CLAUDE_CODE_USE_VOLC_ARK=1 — enable this adapter
 * - ARK_API_URL — default https://ark.cn-beijing.volces.com/api/v3/responses
 * - ARK_API_KEY — optional if x-api-key is already set on the Anthropic request (e.g. ANTHROPIC_API_KEY)
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

/** Turn Anthropic message content into a plain string for Ark `input` items. */
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
    else if (t === 'image' || t === 'document') parts.push(`[${String(t)}]`)
    else parts.push(`[${String(t)}]`)
  }
  return parts.join('\n')
}

type ArkInputMessage = { role: string; content: string }

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
    out.push({
      role: role as string,
      content: flattenAnthropicContent(msg.content),
    })
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
  // Responses API style (non-streaming): output[].content[].text
  const output = o.output
  if (Array.isArray(output) && output.length > 0) {
    const last = output[output.length - 1]
    if (last && typeof last === 'object') {
      const lo = last as Record<string, unknown>
      const content = lo.content
      if (
        Array.isArray(content) &&
        content[0] &&
        typeof content[0] === 'object'
      ) {
        const c0 = content[0] as Record<string, unknown>
        if (typeof c0.text === 'string') return c0.text
      }
    }
  }
  if (typeof o.text === 'string') return o.text
  return null
}

function buildAnthropicMessageJson(text: string, model: string): string {
  const id = `msg_${randomUUID()}`
  const payload = {
    id,
    type: 'message',
    role: 'assistant',
    content: [{ type: 'text', text }],
    model,
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: 0,
      output_tokens: 0,
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
      const anthropicJson = buildAnthropicMessageJson(outText, model)
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
