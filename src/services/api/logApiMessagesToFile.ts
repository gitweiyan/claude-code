import { appendFile } from 'fs/promises'
import { isAbsolute, join } from 'path'
import { getClaudeConfigHomeDir, isEnvTruthy } from '../../utils/envUtils.js'
import { jsonStringify } from '../../utils/slowOperations.js'

/**
 * When CLAUDE_CODE_LOG_API_MESSAGES=1, append each API request's `messages`
 * array (as sent to the model, after cache breakpoints) as one JSON line to a log file.
 *
 * - Default file: ~/.claude/claude.info
 * - Override path: CLAUDE_CODE_LOG_API_MESSAGES_PATH (absolute, or relative to cwd)
 *
 * Security: may contain secrets and PII — do not enable in shared environments.
 */

let appendChain: Promise<void> = Promise.resolve()

export function isApiMessagesFileLoggingEnabled(): boolean {
  return isEnvTruthy(process.env.CLAUDE_CODE_LOG_API_MESSAGES)
}

function resolveLogFilePath(): string {
  const raw = process.env.CLAUDE_CODE_LOG_API_MESSAGES_PATH?.trim()
  if (raw) {
    return isAbsolute(raw) ? raw : join(process.cwd(), raw)
  }
  return join(getClaudeConfigHomeDir(), 'claude.info')
}

export function logApiMessagesToFile(payload: {
  querySource?: string
  model: string
  messages: unknown[]
}): void {
  if (!isApiMessagesFileLoggingEnabled()) return

  const filePath = resolveLogFilePath()
  const record = {
    ts: new Date().toISOString(),
    querySource: payload.querySource,
    model: payload.model,
    messageCount: payload.messages.length,
    messages: payload.messages,
  }
  const line = `${jsonStringify(record)}\n`

  appendChain = appendChain
    .then(() => appendFile(filePath, line, 'utf8'))
    .catch(() => {
      /* avoid breaking API calls if disk is full or path invalid */
    })
}
